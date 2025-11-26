# Rethinking the Architecture Design for Efficient Generic Event  Boundary Detection
[Rethinking the Architecture Design for Efficient Generic Event Boundary Detection](https://arxiv.org/abs/2407.12622)) , ACM MM 2024

[Ziwei Zheng](https://github.com/Ziwei-Zheng), [Zechuan Zhang](https://github.com/zechuan2024), [Yulin Wang](https://github.com/blackfeather-wang), Shiji Song, Gao Huang, [Le Yang](https://github.com/yangle15)
## Overview
In this paper, we experimentally reexamine the architecture of GEBD models and uncover several surprising findings. Firstly, we reveal that a concise GEBD baseline model already achieves promising performance without any sophisticated design. Secondly, we find that the common design of GEBD models using image-domain backbones can contain plenty of architecture redundancy, motivating us to gradually “modernize” each component to enhance efficiency. Thirdly, we show that the GEBD models using image-domain backbones conducting the spatiotemporal learning in a spatial-then-temporal greedy manner can suffer from a distraction issue, which might be the inefficient villain for the GEBD. Using a video-domain backbone to jointly conduct spatiotemporal modeling for GEBD is an effective solution for this issue. 
The outcome of our exploration significantly outperforms the previous SOTA methods under the same backbone choice. 
![fig1](https://github.com/Ziwei-Zheng/EfficientGEBD/blob/main/images/fig1.jpg)
## Getting started

1. git clone our repository, creating a python environment and activate it via the following command:
```
git clone https://github.com/Ziwei-Zheng/EfficientGEBD.git
cd EfficientGEBD
conda create --name EfficientGEBD python=3.10
conda activate EfficientGEBD
pip install -r requirements.txt
```
2. Download [config-files](https://drive.google.com/drive/folders/19cNQaVu3Alxn8VYKJX4FKxl0wEFwCDxI?usp=drive_link) , [CSN-pretrained](https://drive.google.com/drive/folders/1HKg63ArlpYJBisUfELp_Rs0hd7akPpfR?usp=drive_link) (including the configs and the checkpoints of CSN model) and make sure you have the following path: `EfficientGEBD/config-files/` and `EfficientGEBD/CSN-pretrained/`.

## Preparing Kinetics-GEBD/TAPOS/SoccerNet Dataset
Please refer to [GUIDE](https://github.com/ZZC/EfficientGEBD/main/GUIDE.md) for preparing Kinetics-GEBD/TAPOS dataset in the following path:
`EfficientGEBD/data/`
and preparing SoccerNet dataset in the following path:
`EfficientGEBD/EffSoccerNet/data/`

**a) Kinetics-GEBD Dataset** is frequently used in most previous works, which contains 54,691 videos randomly selected from Kinetics-400. You can click [Kinetics-GEBD](https://github.com/StanLei52/GEBD) for more details. Since the annotations for the test set are not publicly accessible, we conducted training on the training set and subsequently evaluated model performance using the validation set.

**b) TAPOS Dataset** 
 we adapt [TAPOS](https://sdolivia.github.io/TAPOS/) by concealing each action instance’s label and conducting experiments based on this modified dataset.

**c) SoccerNet Dataset** 
SoccerNet-v2 is used as an additional benchmark for action spotting detection, which can be viewed as a sub-category of generic event boundaries.


## Training scripts

Training EfficientGEBD with **Kinetics-GEBD** or **TAPOS** via the following command (choose the model you want to train at 'EfficientGEBD/script/train/):
For example, training BasicGEBD-ResNet50-L1:
```
bash script/train/BasicGEBD-ResNet50-L1.sh
```
Training EfficientGEBD-ResNet50-L4:

```
bash script/train/EfficientGEBD-ResNet50-L4.sh
```
While training models with **SoccerNet**, use：
```
bash EffSoccerNet/EffSoccerNet_train.sh
``` 
Note that you should change `DATASETS.TRAIN = ('GEBD_train',)`/`DATASETS.TEST = ('GEBD_minval',)` 
into `DATASETS.TRAIN = ('TAPOS_train',)`/`DATASETS.TEST = ('TAPOS_val',)`,
and `OUTPUT_DIR: 'output/Kinetics-GEBD/'` into `OUTPUT_DIR: 'output/TAPOS/'`
 in `config-files/baseline.yaml`  if you are training with **TAPOS**.
 
## Testing our trained model

We only provide the **checkpoint** that generating our highest score in the paper[[download](https://drive.google.com/file/d/1S4M-xnKpjWFGBimcRYzlEDFhDsWQWF_-/view?usp=drive_link)]. Unzip the specific folder to your project in the following path:'EfficientGEBD/output/Kinetics-GEBD/`$YOUR_UNZIPPED_DIR$`')

| _Method_| _Backbone_  | _F1 score_ | _GFLOPs_ | _FPS_ |
| :----: | :----: | :----: | :----: | :----: |
|  _BasicGEBD_ | ResNet50-L4  | **77.1** | 4.36 | 1562 |
| _BasicGEBD_  | ResNet50-L2  | **76.8** | 2.08 | 2325 |
| _BasicGEBD_  | ResNet34-L4  | **77.0** | 3.92 | 2386 |
| _BasicGEBD_  | ResNet18-L4  | **77.2** | 2.07 |2380  |
| _BasicGEBD_  | ResNet152-L4  | **77.2** | 11.77| 847 |
| _EfficientGEBD_  | ResNet50-L2*  | **78.3** | 2.10 | 2116 |
| _EfficientGEBD_  | ResNet50-L4*  | **78.7** | 4.39 |1541  |
| _EfficientGEBD_  | CSNR152-L2* | **80.6** | 2.00 | 1215 | 
| _EfficientGEBD_  | CSNR152-L4* | **82.9** | 6.40 | 1025 |


Test trained model with **Kinetics-GEBD** or **TAPOS** via the following command (choose the model you want to test at 'EfficientGEBD/script/test/'):
```
bash script/test/BasicGEBD-ResNet50-L1.sh
```

While testing model with **SoccerNet**, use:
```
bash EffSoccerNet/SoccerNet_test.sh
``` 

## Citation

If you find our work helps, please cite our paper:

```
@inproceedings{zheng2024rethinking,
  title={Rethinking the Architecture Design for Efficient Generic Event Boundary Detection},
  author={Zheng, Ziwei and Zhang, Zechuan and Wang, Yulin and Song, Shiji and Huang, Gao and Yang, Le},
  booktitle={Proceedings of the 32nd ACM International Conference on Multimedia},
  year={2024}
}
```

## Contact

If you have any questions, please feel free to contact me at ziwei.zheng@stu.xjtu.edu.cn


## Acknowledgement
