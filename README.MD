# Hand eye calibration circle segmentation

Code of the paper: [A novel hand-eye semi-automatic calibration process for laser profilometers using machine learning](https://www.editorialmanager.com/meas/mainpage.html)



## 1. Introduction

This project is the circle segmentation implementation for robot and machine hand-eye calibration based on the [PointNet-based](https://github.com/charlesq34/pointnet) networks. The project includes:

- Training, evaluation and testing of the [Standard Sphere Seg Dataset V1]().
- Profile files batch segmentation and fitting based on the pretrained networks and RANSAC method.



## 2. Performances

We provide the OA (Overall Accuracy), IoU (Intersection-over-Union), mIoU (Mean Intersection-over-Union) ,differences of the circle middle points ($D_{mp}$) and radius ($D_{r}$) and inference speed of each network on both validation set and test set. The training parameters can be found in the ```train_hand_eye.py```.

Table 1 Performances on the validation set

|  **Model**   |  OA   | IoU0  | IoU0  | mIoU  |
| :----------: | :---: | :---: | :---: | :---: |
| PointNet-512 | 0.993 | 0.997 | 0.993 | 0.995 |
| PointNet-256 | 0.998 | 0.997 | 0.992 | 0.994 |
| PointNet-128 | 0.997 | 0.996 | 0.990 | 0.993 |

Table 2 Performances on the test set

|  **Model**   |  OA   | IoU0  | IoU1  | mIoU  | **$\bold D_{mp}$** (mm) | **$\bold D_{r}$** (mm) | FPS  |
| :----------: | :---: | :---: | :---: | :---: | :---------------------: | :--------------------: | ---- |
| PointNet-512 | 0.991 | 0.988 | 0.967 | 0.978 |          0.026          |         0.011          | 89.3 |
| PointNet-256 | 0.991 | 0.988 | 0.967 | 0.978 |          0.026          |         0.010          | 93.5 |
| PointNet-128 | 0.992 | 0.989 | 0.970 | 0.980 |          0.024          |         0.011          | 95.2 |



## 3. Project file structures

```bash
hand_eye_cali_circle_segmentation
├── checkpoints
│   ├── PointNet_128.pkl 
│   ├── PointNet_256.pkl
│   └── PointNet_512.pkl
├── data
│   └── Standard_sphere_seg_dataset_v1
├── load_plot.py
├── log
├── output_file
├── requirements.txt
├── train_hand_eye.py # training
├── hand_eye_infer.py # evaluation, demo visualization, batch processing
├── ransac_circle_fitting.py # circle batch fitting
├── Naive_cross_line_net.py # networks definetion
├── training_infer_tools
│   └── my_lr_schedule.py
└── utils
    ├── config.py # parameters settings
    └── Handeye_datasets.py # dataset loading and operation 
```



## 4. Dataset

Dataset has been put on the `data`folder: [dataset readme]()



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
python train_hand_eye.py --model Pointnet128 --train_path ./data/Standard_sphere_seg_dataset_v1/train_file.txt

# circle segmentation demo visualization 
python hand_eye_infer.py --mode infer_vis

# test set evaluation
python hand_eye_infer.py --mode infer_results

# test set circle batch segmentation and saving
python hand_eye_infer.py --mode batch_circle_segmentation

# segmented circle fitting
python ransac_circle_fitting.py
```