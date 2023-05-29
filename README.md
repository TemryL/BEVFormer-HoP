<div align="center">   

# Bird’s-Eye View from Monocular Cameras: Group 3
</div>

- Group number: 3
- Group Member:
    1. Mery Tom, **SCIPER: 297217** (tom.mery@epfl.ch)
    1. Charfeddine Ramy, **SCIPER: 295758** (ramy.charfeddine@epfl.ch)

* [Abstract](#abstract)
* [Methods](#methods)
* [Contribution](#contribution)
* [Getting Started](#getting-started)
    * [Instalation](#getting-started)
    * [Dataset](#getting-started)
    * [Run and Eval](#getting-started)
* [Experiments](#experiments)
* [Results](#results)
* [Conclusion](#conclusion)

# Abstract
The objective is to develop a deep learning
model that can transform monocular camera images of the
surrounding into a bird’s eye view map. The output of the
model will be bird’s eye view map that can be used to train
the object detection, tracking and predictions algorithms
of the Tesla Autopilot system.


# Methods
![method](figs/arch.png "model arch")

# Contribution
Very recent paper [8] comes up with a new
paradigm, named Historical Object Prediction
(HoP) for multi-view 3D detection to leverage
temporal information more effectively (fig.2). This
method allows state-of-the-art architectures to perform even better by generating a pseudo BEV
feature map of timestamp (t−k) from its adjacent
frames and utilize this feature to predict the object
set at timestamp (t−k). HoP is performed only during
training and thus, does not introduce extra overheads
during inference. HoP is described as a plug-and-play approach and can be easily incorporated into
state-of-the-art BEV detection frameworks including BEVFormer [6].
As the paper is very recent, the implementation is
yet not available and therefore the main contribution
of the project is the implementation of the proposed
HoP method including it to the BEVFormer model.

# Getting Started
- [Installation](docs/install.md) 
- [Dataset](docs/dataset.md)
- [Run and Eval](docs/run.md)

# Experiments

# Results
| Backbone | Method | Lr Schd | NDS| mAP|memroy | Config | Download |
| :---: | :---: | :---: | :---: | :---:|:---:| :---: | :---: |
| R50 | BEVFormer-tiny_fp16 | 24ep | 35.9|25.7 | - |[config](projects/configs/bevformer_fp16/bevformer_tiny_fp16.py) |[model](https://github.com/zhiqi-li/storage/releases/download/v1.0/bevformer_tiny_fp16_epoch_24.pth)/[log](https://github.com/zhiqi-li/storage/releases/download/v1.0/bevformer_tiny_fp16_epoch_24.log) |
| R50 | BEVFormer-tiny | 24ep | 35.4|25.2 | 6500M |[config](projects/configs/bevformer/bevformer_tiny.py) |[model](https://github.com/zhiqi-li/storage/releases/download/v1.0/bevformer_tiny_epoch_24.pth)/[log](https://github.com/zhiqi-li/storage/releases/download/v1.0/bevformer_tiny_epoch_24.log) |
| [R101-DCN](https://github.com/zhiqi-li/storage/releases/download/v1.0/r101_dcn_fcos3d_pretrain.pth)  | BEVFormer-small | 24ep | 47.9|37.0 | 10500M |[config](projects/configs/bevformer/bevformer_small.py) |[model](https://github.com/zhiqi-li/storage/releases/download/v1.0/bevformer_small_epoch_24.pth)/[log](https://github.com/zhiqi-li/storage/releases/download/v1.0/bevformer_small_epoch_24.log) |
| [R101-DCN](https://github.com/zhiqi-li/storage/releases/download/v1.0/r101_dcn_fcos3d_pretrain.pth)  | BEVFormer-base | 24ep | 51.7|41.6 |28500M |[config](projects/configs/bevformer/bevformer_base.py) | [model](https://github.com/zhiqi-li/storage/releases/download/v1.0/bevformer_r101_dcn_24ep.pth)/[log](https://github.com/zhiqi-li/storage/releases/download/v1.0/bevformer_r101_dcn_24ep.log) |

# Conclusion



