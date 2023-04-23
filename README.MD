# Hand eye calibration circle segmentation.

Code of the paper: [A novel hand-eye semi-automatic calibration process for laser profilometers using machine learning](https://www.editorialmanager.com/meas/mainpage.html)



## Changelog

- **Apr 22, 2023**: Paper has been accepted by *Measurement.*
- **March 12, 2023**: Add PointNet-64 relative training code, performance, and relative checkpoint.
- **March 2, 2023**: Add *hough_circle_fitting.py*, code implementation of Hough Transform method.
- **Feb 25, 2023**: Add *RLTS_WRLTS_circle_fitting.py*, code implementation of RLTS and WRLTS circle fitting methods.



## 1. Introduction

This repository is the circle segmentation implementation for robot and machine hand-eye calibration based on the [PointNet-based](https://github.com/charlesq34/pointnet) networks. The repository includes:

- Training, evaluation and testing code, checkpoints of the [Standard Sphere Seg Dataset V1](./data/Standard_sphere_seg_dataset_v1/).
- Profile files batch segmentation and fitting based on the pretrained networks and RANSAC method.
- Performance comparison of circle segmentation methods, including PointNet-based networks, Hough transform, WRLTS, and RANSAC methods.



## 2. Performances

We provide the OA (Overall Accuracy), IoU (Intersection-over-Union), mIoU (Mean Intersection-over-Union) ,differences of the circle middle points ($D_{mp}$) and radius ($D_{r}$) and inference speed of each network on both validation set and test set. The training parameters can be found in the ```train_hand_eye.py```.

Table 1 Performances on the validation set

|  **Model**   |  OA   | IoU0  | IoU0  | mIoU  |
| :----------: | :---: | :---: | :---: | :---: |
| PointNet-512 | 0.993 | 0.997 | 0.993 | 0.995 |
| PointNet-256 | 0.998 | 0.997 | 0.992 | 0.994 |
| PointNet-128 | 0.997 | 0.996 | 0.990 | 0.993 |
| PointNet-64  | 0.997 | 0.995 | 0.987 | 0.991 |

Table 2 Performances on the test set

|  **Model**   |  OA   | IoU0  | IoU1  | mIoU  | $D_{mp}$ (mm) | $D_{r}$ (mm) | FPS  |
| :----------: | :---: | :---: | :---: | :---: | :---------------------: | :--------------------: | ---- |
| PointNet-512 | 0.991 | 0.988 | 0.967 | 0.978 |          0.026          |         0.011          | 89.3 |
| PointNet-256 | 0.991 | 0.988 | 0.967 | 0.978 |          0.026          |         0.010          | 93.5 |
| PointNet-128 | 0.992 | 0.989 | 0.970 | 0.980 |          0.024          |         0.011          | 95.2 |
| PointNet-64 | 0.992 | 0.989 | 0.971 | 0.980 | 0.022 | 0.010 | 96.1 |



## 3. Project file structures

```bash
hand_eye_cali_circle_segmentation
├── checkpoints # pre-trained network files
│   ├── PointNet_64.pkl 
│   ├── PointNet_128.pkl 
│   ├── PointNet_256.pkl
│   └── PointNet_512.pkl
├── data
│   └── Standard_sphere_seg_dataset_v1
├── log
├── output_file
├── requirements.txt # environmentrequirements
├── train_hand_eye.py # training
├── hand_eye_infer.py # evaluation, demo visualization, batch processing
├── ransac_circle_fitting.py # circle batch fitting
├── RLTS_WRLTS_circle_fitting.py # RLTS and WRLTS circle fitting method 
├── hough_circle_fitting.py.py # Hough Transform circle detection method
├── Naive_cross_line_net.py # networks definetion
├── training_infer_tools
│   └── my_lr_schedule.py
└── utils
    ├── config.py # parameters settings
    └── Handeye_datasets.py # dataset loading and operation 
```



## 4. Dataset

Dataset has been put on the `data`folder: [dataset readme](./data/Standard_sphere_seg_dataset_v1/README.md)



## 5. Dependencies installation and demo running

Here is an example of creating environment from scratch with `anaconda`

```sh
# create conda env
conda create --name he_seg python=3.8.5
conda activate he_seg

# install pytorch
conda install pytorch torchvision cudatoolkit=10.2 -c pytorch

# install other dependancy
pip install -r requriements.txt

# start visdom
python -m visdom.server #Visdom need to be started before training for metrics visualization

```

The project has been tested on the Ubuntu 18.04, cuda 11.0 and Pytorch 1.7.1 with NVIDIA RTX3080 GPU and AMD 3700X @3.6 GHz CPU.

Before running demos, some parameters need to be set in the `config.py`.

And we also provide training, evaluation and circle fitting demos as:

```bash
# training
python train_hand_eye.py --help #specify the parameters before running
python train_hand_eye.py --model Pointnet128 --train_path ./data/Standard_sphere_seg_dataset_v1/train_file.txt #performing training code

# circle segmentation demo visualization 
python hand_eye_infer.py --mode infer_vis

# test set evaluation
python hand_eye_infer.py --mode infer_results

# test set circle batch segmentation and saving, related to the section 4.4.5 of the paper
python hand_eye_infer.py --mode batch_circle_segmentation

# segmented circle fitting, related to the section 4.4.5 of the paper, parameters can be set in config.py
python ransac_circle_fitting.py

#circle detection using Hough Transform method (this might take a very long time)
python hough_circle_fitting.py

```



## 6. Citations

If you use our code or data in your research, please cite our paper:

```
@InProceedings{swinunet,
author = {},
title = {},
booktitle = {},
year = {}
}
```