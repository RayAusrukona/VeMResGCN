# This is for OU-MVLP

## Preparation
Clone the repository and install the dependencies from `requirements.txt`.

### Dataset
- OUMVLP-Pose: Download from [here](http://www.am.sanken.osaka-u.ac.jp/BiometricDB/GaitLPPose.html).


## Running the code
We use [PyTorch Lightning CLI](https://pytorch-lightning.readthedocs.io/en/stable/common/lightning_cli.html) for configuration and training.

Train:
```bash
# OUMVLP-Pose (OpenPose)
python3 gaitgraph_oumvlp.py fit --config configs/oumvlp.yaml
# OUMVLP-Pose (AlphaPose)
python3 gaitgraph_oumvlp.py fit --config configs/oumvlp.yaml --data.keypoints alphapose
```

Test:
```bash
python3 gaitgraph_{casia_b,oumvlp}.py predict --config <path_to_config_file> --ckpt_path <path_to_checkpoint> --model.tta True
```

Logs and checkpoints will be saved to `lighting_logs` and can be shown in tensorboard with:
```bash
tensorboard --logdir lightning_logs
```
