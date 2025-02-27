import sys
import time
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from ray import tune
from ray.tune.schedulers import HyperBandScheduler

from datasets import dataset_factory
from datasets.augmentation import *
from datasets.graph import Graph
from evaluate import evaluate, _evaluate_casia_b
from losses import SupConLoss

from common import *
from utils import AverageMeter


def train(train_loader, model, criterion, optimizer, scheduler, scaler, epoch, opt):
    """one epoch training"""
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    end = time.time()

    total_correct = 0

    for idx, (points, target) in enumerate(train_loader):
        data_time.update(time.time() - end)
        labels = target[0]
        ########################## view ######################################
        view = target[3]
        view = [int(int(i)/18) for i in view]
        #print("view: ", view)
        view = torch.tensor(view)
        #######################################################################

        if torch.cuda.is_available():
            points = points.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
            view = view.cuda(non_blocking=True)

        bsz = labels.shape[0]

        with torch.cuda.amp.autocast(enabled=opt.use_amp):
            features, angle_probe = model(points)            # 256, 128; 256, 11
            f1 = features
            f2 = features
            features_sup = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
            loss = criterion(features_sup, labels)

            cross_loss = nn.CrossEntropyLoss()
            ce_loss = cross_loss(angle_probe, view)
            _, angle= torch.max(angle_probe, 1)

            correct = (angle == view).sum()
            total_correct += correct

            total_correct = 0

            loss = loss + 0.01*ce_loss


        # update metric
        losses.update(loss.item(), bsz)

        # SGD
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scheduler.step()
        scaler.update()
        optimizer.zero_grad()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if (idx + 1) % opt.log_interval == 0:
            print(
                f"Train: [{epoch}][{idx + 1}/{len(train_loader)}]\t"
                f"BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                f"DT {data_time.val:.3f} ({data_time.avg:.3f})\t"
                f"loss {losses.val:.3f} ({losses.avg:.3f})"
            )
            sys.stdout.flush()

    return losses.avg


def main(opt):
    opt = setup_environment(opt)
    graph = Graph("coco")       #get 3D adjacency matrix 4x17x17, 4 = valid hop

    # Dataset
    transform = transforms.Compose(
        [
            MirrorPoses(opt.mirror_probability),  #0.5
            FlipSequence(opt.flip_probability),   #0.5
            RandomSelectSequence(opt.sequence_length),   #60 sequence randomly nibe
            ShuffleSequence(opt.shuffle),    #false
            PointNoise(std=opt.point_noise_std),   #0.05, std = 0.15
            JointNoise(std=opt.joint_noise_std),   #0.1, std = 0.5
            MultiInput(graph.connect_joint, opt.use_multi_branch),  #connect_joint = np.array([5,0,0,1,2,0,0,5,6,7,8,5,6,11,12,13,14]), multi_branch = false
            ToTensor()  #convert image to tensor
        ],
    )

    dataset_class = dataset_factory(opt.dataset)   # create dataset using dataset_factory combining different directories, file systems, or file formats.
    dataset = dataset_class(       #  pass the info into object named dataset_class
        opt.train_data_path,       # casia-b_pose_train_valid.csv  start 1 to 74
        train=True,
        sequence_length=opt.sequence_length,
        transform = transforms.Compose(
        [
            MirrorPoses(opt.mirror_probability),  #0.5
            FlipSequence(opt.flip_probability),   #0.5
            RandomSelectSequence(opt.sequence_length),   #60 sequence randomly nibe
            ShuffleSequence(opt.shuffle),    #false
            PointNoise(std=opt.point_noise_std),   #0.05, std = 0.15
            JointNoise(std=opt.joint_noise_std),   #0.1, std = 0.5
            MultiInput(graph.connect_joint, opt.use_multi_branch),  #connect_joint = np.array([5,0,0,1,2,0,0,5,6,7,8,5,6,11,12,13,14]), multi_branch = false
            ToTensor()  #convert image to tensor
        ],
    )

    )

    dataset_valid = dataset_class(
        opt.valid_data_path,       #  casia-b_pose_valid.csv    start 60, finish 74
        sequence_length=opt.sequence_length,
        transform=transforms.Compose(
            [
                SelectSequenceCenter(opt.sequence_length),     # sequence length = 60
                MultiInput(graph.connect_joint, opt.use_multi_branch),     # return data new 3,6,60,17
                ToTensor()
            ]
        ),
    )

    train_loader = torch.utils.data.DataLoader(     # import and export data
        dataset,
        batch_size=opt.batch_size,
        num_workers=opt.num_workers,
        pin_memory=True,
        shuffle=True,            ################## shuffle on batch
    )

    val_loader = torch.utils.data.DataLoader(
        dataset_valid,
        batch_size=opt.batch_size_validation,    # 256
        num_workers=opt.num_workers,
        pin_memory=True,
    )

    # Model & criterion
    model, model_args = get_model_resgcn(graph, opt)   # return with all blocks info

    criterion = SupConLoss(temperature=opt.temp)   # calculate the loss

    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model, opt.gpus)

    if opt.cuda:      # enter here
        model.cuda()
        criterion.cuda()      #  access cuda

    # Trainer
    optimizer, scheduler, scaler = get_trainer(model, opt, len(train_loader))   # casia-b_pose_train_valid.csv  start 1  # make training efficiently

    # Load checkpoint or weights
    load_checkpoint(model, optimizer, scheduler, scaler, opt)

    # Tensorboard
    writer = SummaryWriter(log_dir=opt.tb_path)   # it is used to visualize the model by tensorboard and create event file

    best_acc = 0
    loss = 0
    for epoch in range(opt.start_epoch, opt.epochs + 1):
        # train for one epoch
        time1 = time.time()
        loss = train(
            train_loader, model, criterion, optimizer, scheduler, scaler, epoch, opt
        )  # return losses.avg = 0.8358

        time2 = time.time()
        # tensorboard logger
        writer.add_scalar("loss/train", loss, epoch)
        writer.add_scalar("learning_rate", optimizer.param_groups[0]["lr"], epoch)

        # evaluation
        result, accuracy_avg, sub_accuracies, dataframe = evaluate(
            val_loader, model, opt.evaluation_fn, use_flip=True           # opt.evaluation_fn = this means evaluation happened on casia_b
        )

        writer.add_text("accuracy/validation", dataframe.to_markdown(), epoch)
        writer.add_scalar("accuracy/validation", accuracy_avg, epoch)
        for key, sub_accuracy in sub_accuracies.items():                         # key = NM#5-6, sub_accuracy = .0696
            writer.add_scalar(f"accuracy/validation/{key}", sub_accuracy, epoch) #       BG#1-2               = .0678
                                                                                 #       CL#1-2               = .0621
        is_best = accuracy_avg > best_acc
        if is_best:
            best_acc = accuracy_avg

        if opt.tune:
            tune.report(accuracy=accuracy_avg)

        if epoch % opt.save_interval == 0 or (is_best and epoch > opt.save_best_start * opt.epochs):
            save_file = os.path.join(opt.save_folder, f"ckpt_epoch_{'best' if is_best else epoch}.pth")
            save_model(model, optimizer, scheduler, scaler, opt, opt.epochs, save_file)

    # save the last model
    save_file = os.path.join(opt.save_folder, "last.pth")
    #print(save_file,"   sssss")
    save_model(model, optimizer, scheduler, scaler, opt, opt.epochs, save_file)

    log_hyperparameter(writer, opt, best_acc, loss)



def _inject_config(config):
    opt_new = {k: config[k] if k in config.keys() else v for k, v in vars(opt).items()}
    main(argparse.Namespace(**opt_new))


def tune_():
    hyperband = HyperBandScheduler(metric="accuracy", mode="max")

    analysis = tune.run(
        _inject_config,
        config={},
        stop={"accuracy": 0.90, "training_iteration": 100},
        resources_per_trial={"gpu": 1},
        num_samples=10,
        scheduler=hyperband
    )

    df = analysis.results_df


if __name__ == "__main__":
    import datetime

    opt = parse_option()  # shuffle and multi_branch = false

    date = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    opt.model_name = f"{date}_{opt.dataset}_{opt.network_name}" \
                     f"_lr_{opt.learning_rate}_decay_{opt.weight_decay}_bsz_{opt.batch_size}"   # 6 iterration

    if opt.exp_name:   # no entry
        opt.model_name += "_" + opt.exp_name

    #sssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssss
    opt.model_path = f"../save/{opt.dataset}_models" #checkpoints
    opt.tb_path = f"../save/{opt.dataset}_tensorboard/{opt.model_name}"  # tensorboard file path contains matrix type

    opt.save_folder = os.path.join(opt.model_path, opt.model_name) # add those two paths
    if not os.path.isdir(opt.save_folder):  #check wheather this directory path exist or not
        os.makedirs(opt.save_folder)         # then make directory
    #sssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssss

    opt.evaluation_fn = None
    if opt.dataset == "casia-b":
        opt.evaluation_fn = _evaluate_casia_b

    if opt.tune:   # here, tune is false
        tune_()
    else:
        main(opt)

