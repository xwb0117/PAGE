# Pre-defined Keypoints Promote Category-level Articulation Pose Estimation via Multi-Modal Alignment
This repository is the official implementation of the paper Pre-defined Keypoints Promote Category-level Articulation Pose Estimation via
Multi-Modal Alignment.

## Installation
This code has been tested on Ubuntu 20.04 with Python 3.7, CUDA 11.3 and Pytorch 1.10.0 Firstly, create environment and install the dependencies:

```bash
conda create -n PAGE python=3.7
conda activate PAGE
pip install -r requirements.txt
```
And build the CUDA kernels for PointNet++
```bash
git clone --recursive https://github.com/erikwijmans/Pointnet2_PyTorch
cd Pointnet2_PyTorch
pip install -r requirements.txt
pip install -e 
```

## Datasets
Download our generated dataset from ArtImage at [BaiduYun(code:o2ou)](https://pan.baidu.com/s/1vyEL3uvvaT1dNvpZc4jyIg?pwd=o2ou) and [OneDrive](https://onedrive.live.com/?cid=0CEC31BD19A86C5F&id=CEC31BD19A86C5F%21s618290dc5a61431fa6b3ced1732bde45&parId=root&o=OneUp), and save in ```/data```

## Training of KP_Estimator

Take Laptop as example:

```bash
python train_KP_Estimator.py --keypointsNo=3  --category=laptop --nparts=2 --part_num=0 --dname=Art
```

## Training of KP_Estimator

```bash
python eval_KP_Estimator.py --keypointsNo=3  --category=laptop --nparts=2 --part_num=0 --dname=Art
```

## Training of PAGE_Estimator

Take Laptop as example:

```bash
python main_PAGE.py --mode=train  --dname=Art  --batch_size=8 --class_name=laptop  --kpt_class=KP  --num_classes=2  --n_sample_points=2048  --input_channel=0  --model=PAGENet  --kpt_num=0  --part_num=0
```

## Training of PAGE_Estimator

```bash
python main_PAGE.py --mode=test  --dname=Art  --batch_size=8 --class_name=laptop  --kpt_class=KP  --num_classes=2  --n_sample_points=2048  --input_channel=0  --model=PAGENet  --kpt_num=0  --part_num=0 --using_ckpts=True
```
