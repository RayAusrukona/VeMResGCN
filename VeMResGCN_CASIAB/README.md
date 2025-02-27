# This is for CASIA-B


## Quick Start

### Prerequisites
- Python >= 3.6
- CUDA >= 10

First, create a virtual environment or install dependencies directly with:
```shell
pip3 install -r requirements.txt
```

### Data preparation
The extraction of the pose data from CASIA-B can either run the commands bellow or download the preprocessed data using:
```shell
cd data
sh ./download_data.sh
```

Optional:
If you choose to run the preprocessing, [download](http://www.cbsr.ia.ac.cn/english/Gait%20Databases.asp) the dataset and run the following commands.
```shell
# Download required weights
cd models
sh ./download_weights.sh

# Copy extraction script
# <PATH_TO_CASIA-B> should be something like: /home/ ... /datasets/CASIA_Gait_Dataset/DatasetB
cd ../data
cp extract_frames.sh <PATH_TO_CASIA-B>

cd <PATH_TO_CASIA-B>
mkdir frames
sh extract_frames.sh
cd frames
find . -type f -regex ".*\.jpg" -print | sort | grep -v bkgrd > ../casia-b_all_frames.csv
cp ../casia-b_all_frames.csv <PATH_TO_REPO>/data

cd <PATH_TO_REPO>/src
export PYTHONPATH=${PWD}:$PYTHONPATH

cd preparation
python3 prepare_detection.py <PATH_TO_CASIA-B> ../../data/casia-b_all_frames.csv ../../data/casia-b_detections.csv
python3 prepare_pose_estimation.py  <PATH_TO_CASIA-B> ../../data/casia-b_detections.csv ../../data/casia-b_pose_coco.csv
python3 split_casia-b.py ../../data/casia-b_pose_coco.csv --output_dir ../../data
```

### Train
To train the model you can run the `train.py` script. To see all options run:
```shell
cd src
export PYTHONPATH=${PWD}:$PYTHONPATH

python3 train.py --help
```

Check `experiments/1_train_*.sh` to see the configurations used in the paper. 

Optionally start the tensorboard with: 
```shell
tensorboard --logdir=save/casia-b_tensorboard 
```

### Evaluation
Evaluate the models using `evaluate.py` script. To see all options run:
```shell
python3 evaluate.py --help
```


## Acknowledgement

The following parts of the code are borrowed from other projects. Thanks for their wonderful work!
- GaitGraph: [tteepe/GaitGraph](https://github.com/tteepe/GaitGraph/tree/main)
- Object Detector: [eriklindernoren/PyTorch-YOLOv3](https://github.com/eriklindernoren/PyTorch-YOLOv3)
- Pose Estimator: [HRNet/HRNet-Human-Pose-Estimation](https://github.com/HRNet/HRNet-Human-Pose-Estimation)
- ST-GCN Model: [yysijie/st-gcn](https://github.com/yysijie/st-gcn)
- ResGCNv1 Model: [yfsong0709/ResGCNv1](https://github.com/yfsong0709/ResGCNv1)
- SupCon Loss: [HobbitLong/SupContrast](https://github.com/HobbitLong/SupContrast)
