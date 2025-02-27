import os
#os.environ['CUDA_VISIBLE_DEVICES'] = "1,2"
import pandas as pd
import pytorch_lightning as pl
import torch


from pytorch_lightning import Trainer
#from pytorch_lightning.callbacks.stochastic_weight_avg


from pytorch_lightning.utilities.cli import LightningCLI
from pytorch_metric_learning import losses, distances
from torch.utils.data import DataLoader
from torch_geometric.data import Data
from torchvision.transforms import Compose

from logo import cli_logo
from datasets.graph import Graph
from datasets.oumvlp_pose import OUMVLPPose
from models import ResGCN, StGCN
from transforms import ToFlatTensor
from transforms.augmentation import RandomSelectSequence, PadSequence, SelectSequenceCenter, NormalizeEmpty, \
    RandomFlipLeftRight, PointNoise, RandomFlipSequence, JointNoise, RandomMove, ShuffleSequence
from transforms.multi_input import MultiInput


############################### SWA ##############################################
from pytorch_lightning.callbacks import StochasticWeightAveraging
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, StochasticWeightAveraging

swa_callback = StochasticWeightAveraging(
    swa_epoch_start=0.8,
    annealing_epochs=10,
    annealing_strategy='cos',
    swa_lrs=[0.0005] # set the swa_lrs key to a list of learning rates
)


lr_monitor_callback = LearningRateMonitor(log_momentum=False, logging_interval='step')
model_checkpoint_callback = ModelCheckpoint(
    filename='gaitgraph-oumvlp-{epoch:02d}-{val_loss_epoch:.2f}',
    save_top_k=1,
    monitor='val_loss_epoch',
    mode='min',
    save_last=True,
    save_weights_only=False
)

########################################################################################



class GaitGraphOUMVLP(pl.LightningModule):
    def __init__(
            self,
            learning_rate: float = 0.005,
            lr_div_factor: float = 25.0,
            loss_temperature: float = 0.01,
            embedding_layer_size: int = 128,
            multi_input: bool = True,
            backend_name="resgcn-n51-r4",
            tta: bool = True
    ):
        super().__init__()
        self.save_hyperparameters()

        self.graph = Graph("oumvlp")
        model_args = {
            "A": torch.tensor(self.graph.A, dtype=torch.float32, requires_grad=False),
            "num_class": embedding_layer_size,
            "num_input": 3 if multi_input else 1,
            "num_channel": 5 if multi_input else 3,
            "parts": self.graph.parts,
        }
        if backend_name == "st-gcn":
            self.backbone = StGCN(3, self.graph, embedding_layer_size=embedding_layer_size)
        else:
            self.backbone = ResGCN(backend_name, **model_args)

        self.distance = distances.LpDistance()

        ###################################################################
        self.csloss = torch.nn.CrossEntropyLoss()
        ###################################################################

        self.train_loss = losses.SupConLoss(loss_temperature, distance=self.distance)
        self.val_loss = losses.ContrastiveLoss(distance=self.distance)

    def forward(self, x):
        return self.backbone(x)[0]

    def training_step(self, batch, batch_idx):
        x, y, _ = batch

        ########################## view ######################################
        view = batch[2][0]

        view = [int(int(i) / 15) for i in view]
        # print("view before ",view)
        cnt = 0
        for i in view:
            if i > 6:
                # print("view2 before ", i)
                view[cnt] = int(i - 5)
                # print("view2 ", view[cnt])
            cnt = cnt + 1

        # print("view: ", view.size())
        # print("view after ", view)
        labels = torch.tensor(view).cuda().long()

        #######################################################################

        # y_hat= self(x)
        y_hat, x, angle_probe = self.backbone(x)
        # print(angle_probe,angle_probe)

        ce_loss = self.csloss(angle_probe, labels)
        _, angle = torch.max(angle_probe, 1)

        """anglet = torch.tensor(angle)
        viewt = torch.tensor(view)

        anglet = anglet.to(0)
        viewt = viewt.to(0)

        total_correct = 0
        correct = (anglet == viewt).sum()
        total_correct += correct

        print("Total correct:", total_correct)"""

        t_loss = self.train_loss(y_hat, y.squeeze())  # sup con loss

        # self.log("train_loss", t_loss, on_epoch=True)
        # self.log("cross_loss", ce_loss, on_epoch=True)

        self.log("train_loss", t_loss, on_epoch=True, prog_bar=True, on_step=False)  # newly added
        self.log("cross_loss", ce_loss, on_epoch=True, prog_bar=True, on_step=False)  # newly added

        total_loss = t_loss + ce_loss * 0.01
        # print("total loss: ", total_loss)
        ########################################################################
        return total_loss

    def validation_step(self, batch, batch_idx):
        x, y, _ = batch
        y_hat = self(x)

        loss = self.val_loss(y_hat, y.squeeze())
        self.log("val_loss", loss, on_step=True)

    def predict_step(self, batch, batch_idx: int, dataloader_idx=None):
        x, y, (angle, seq_num, _) = batch
        feature = self.backbone(x)[1]

        return feature, x, y, angle, seq_num

    def test_step(self, batch, batch_idx):
        x, y, (angle, seq_num, _) = batch
        bsz = x.shape[0]

        if self.hparams.tta:
            multi_input = MultiInput(self.graph.connect_joint, self.graph.center, self.hparams.multi_input)
            x_flipped = torch.stack([
                multi_input(Data(x=d[:, :, 0, :3].flip(0), device=x.device)).x for d in x
            ])
            x_lr_flipped = torch.stack([
                multi_input(Data(x=d[:, self.graph.flip_idx, 0, :3], device=x.device)).x for d in x
            ])

            x = torch.cat([x, x_flipped, x_lr_flipped], dim=0)

        y_hat = self(x)

        if self.hparams.tta:
            f1, f2, f3 = torch.split(y_hat, [bsz, bsz, bsz], dim=0)
            y_hat = torch.cat((f1, f2, f3), dim=1)

        return y_hat, y, angle, seq_num

    def test_epoch_end(self, outputs, print_output=True):
        embeddings = dict()
        for batch in outputs:
            y_hat, subject_id, angle, seq_num = batch
            embeddings.update({
                (subject_id[i].item(), angle[i].item(), seq_num[i].item()): y_hat[i]
                for i in range(y_hat.shape[0])
            })

        angles = list(range(0, 91, 15)) + list(range(180, 271, 15))
        num_angles = len(angles)
        gallery = {k: v for (k, v) in embeddings.items() if k[2] == 0}

        gallery_per_angle = {}
        for angle in angles:
            gallery_per_angle[angle] = {k: v for (k, v) in gallery.items() if k[1] == angle}

        probe = {k: v for (k, v) in embeddings.items() if k[2] == 1}

        accuracy = torch.zeros((num_angles + 1, num_angles + 1))
        correct = torch.zeros_like(accuracy)
        total = torch.zeros_like(accuracy)

        for gallery_angle in angles:
            gallery_embeddings = torch.stack(list(gallery_per_angle[gallery_angle].values()), 0)
            gallery_targets = list(gallery_per_angle[gallery_angle].keys())
            gallery_pos = angles.index(gallery_angle)

            probe_embeddings = torch.stack(list(probe.values()))
            q_g_dist = self.distance(probe_embeddings, gallery_embeddings)

            for idx, target in enumerate(probe.keys()):
                subject_id, probe_angle, _ = target
                probe_pos = angles.index(probe_angle)

                min_pos = torch.argmin(q_g_dist[idx])
                min_target = gallery_targets[int(min_pos)]

                if min_target[0] == subject_id:
                    correct[gallery_pos, probe_pos] += 1
                total[gallery_pos, probe_pos] += 1

        accuracy[:-1, :-1] = correct[:-1, :-1] / total[:-1, :-1]

        accuracy[:-1, -1] = torch.mean(accuracy[:-1, :-1], dim=1)
        accuracy[-1, :-1] = torch.mean(accuracy[:-1, :-1], dim=0)

        accuracy_avg = torch.mean(accuracy[:-1, :-1])
        accuracy[-1, -1] = accuracy_avg
        self.log("test/accuracy", accuracy_avg)

        for angle, avg in zip(angles, accuracy[:-1, -1].tolist()):
            self.log(f"test/probe_{angle}", avg)
        for angle, avg in zip(angles, accuracy[-1, :-1].tolist()):
            self.log(f"test/gallery_{angle}", avg)

        df = pd.DataFrame(
            accuracy.numpy(),
            angles + ["mean"],
            angles + ["mean"],
        )
        df = (df * 100).round(1)

        if print_output:
            print(f"accuracy: {accuracy_avg * 100:.1f} %")
            print(df.to_markdown())
            print(df.to_latex())

        return df

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.learning_rate
        )
        lr_schedule = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            self.hparams.learning_rate,
            epochs=self.trainer.max_epochs,
            steps_per_epoch=len(self.trainer.datamodule.train_dataloader())
        )
        lr_dict = {
            "scheduler": lr_schedule,
            "interval": "step"
        }
        return [optimizer], [lr_dict]


##########################################################################################
class CheckpointEveryNSteps(pl.Callback):

    def __init__(
        self,
        save_step_frequency=175,
        prefix="N-Step-Checkpoint",
        lightning_logs= False,
    ):

        self.save_step_frequency = save_step_frequency
        self.prefix = prefix
        self.lightning_logs = lightning_logs

    def on_train_batch_end(self,trainer: "pl.Trainer",pl_module: "pl.LightningModule",_,batch: 512,batch_idx: int,unused: int = 0, ) -> None:       # newly added

    #def on_train_batch_end(self, trainer: pl.Trainer, _):   # previous
        epoch = trainer.current_epoch
        global_step = trainer.global_step
        if global_step % self.save_step_frequency == 0:
            if self.lightning_logs:
                filename = trainer.checkpoint_callback.filename
            else:
                filename = f"{self.prefix}_{epoch:}_{global_step:}.ckpt"
            ckpt_path = os.path.join(trainer.checkpoint_callback.dirpath, filename)
            trainer.save_checkpoint(ckpt_path)
############################################################################################




class OUMVLPPoseModule(pl.LightningDataModule):
    def __init__(
            self,
            data_path: str = "/data",
            dataset_path: str = "/OUMV_skeleton_dataset",
            keypoints: str = "alphapose",
            batch_size: int = 768,
            num_workers: int = 8,
            sequence_length: int = 30,
            multi_input: bool = True,
            flip_sequence_p: float = 0.5,
            flip_lr_p: float = 0.5,
            joint_noise: float = 0.1,
            point_noise: float = 0.05,
            random_move: (float, float) = (3, 1),
            train_shuffle_sequence: bool = False,
            test_shuffle_sequence: bool = False,
            confidence_noise: float = 0.01,
    ):
        super().__init__()
        self.graph = Graph("oumvlp")

        transform_train = Compose([
            PadSequence(sequence_length),
            RandomFlipSequence(flip_sequence_p),
            RandomSelectSequence(sequence_length),
            ShuffleSequence(train_shuffle_sequence),
            RandomFlipLeftRight(flip_lr_p, flip_idx=self.graph.flip_idx),
            JointNoise(joint_noise),
            PointNoise(point_noise),
            RandomMove(random_move),
            MultiInput(self.graph.connect_joint, self.graph.center, enabled=multi_input),
            ToFlatTensor()
        ])
        transform_val = Compose([
            NormalizeEmpty(),
            PadSequence(sequence_length),
            SelectSequenceCenter(sequence_length),
            ShuffleSequence(test_shuffle_sequence),
            MultiInput(self.graph.connect_joint, self.graph.center, enabled=multi_input),
            ToFlatTensor()
        ])

        self.dataset_train = OUMVLPPose(data_path, dataset_path, keypoints, "train", transform=transform_train)
        self.dataset_val = OUMVLPPose(data_path, dataset_path, keypoints, "test", transform=transform_val)
        self.dataset_test = OUMVLPPose(data_path, dataset_path, keypoints, "test", transform=transform_val)

        self.batch_size = batch_size
        self.num_workers = num_workers

    def train_dataloader(self):
        return DataLoader(self.dataset_train, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.dataset_val, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.dataset_test, batch_size=self.batch_size, num_workers=self.num_workers)

    def predict_dataloader(self):
        return DataLoader(self.dataset_test, batch_size=self.batch_size, num_workers=self.num_workers)


"""def cli_main():
    LightningCLI(
        GaitGraphOUMVLP,
        OUMVLPPoseModule,
        seed_everything_default=5318008,
        save_config_overwrite=True,
        run=True
    )"""

def cli_main():
    cli= GaitGraphOUMVLP()#.load_from_checkpoint("")

    trainer = Trainer(accelerator="gpu", devices=[0], max_epochs=950, num_processes=1, overfit_batches=0.0,
                      track_grad_norm=-1,check_val_every_n_epoch=10,max_steps=-1,limit_train_batches=1.0,limit_val_batches=1.0,limit_test_batches=1.0,
                      limit_predict_batches=1.0,
                      val_check_interval=1.0,log_every_n_steps=25,precision=32,num_sanity_val_steps=2,reload_dataloaders_every_n_epochs=0,
                      callbacks=[lr_monitor_callback, model_checkpoint_callback, swa_callback])


    trainer.fit(cli, OUMVLPPoseModule())
    #trainer.test(cli,dataloaders=OUMVLPPoseModule(),ckpt_path="")


if __name__ == "__main__":
    cli_logo()
    cli_main()
