# Bird’s-Eye View from Monocular Cameras: Group 3

- Group number: 3
- Group Member:
    1. Mery Tom, **SCIPER: 297217** (tom.mery@epfl.ch)
    1. Charfeddine Ramy, **SCIPER: 295758** (ramy.charfeddine@epfl.ch)

* [Abstract](#abstract)
* [Methods](#methods)
* [Contribution](#contribution)
* [Getting Started](#getting-started)
    * [Installation](#getting-started)
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
of the Tesla Autopilot system. Detailed introduction, litterature review and problem statement are available in `milestone1.pdf`.


# Methods
> **BEVFormer: Learning Bird's-Eye-View Representation from Multi-Camera Images via Spatiotemporal Transformers**, ECCV 2022
> - [Paper in arXiv](http://arxiv.org/abs/2203.17270), 13 Jul 2022

BEVFormer models have been shown to
achieve state-of-the-art performance on a variety of
benchmarks for BEV map generation in autonomous
driving applications. This model consists of a convolutional
neural network (CNN) backbone that extracts
features from the input images and a
transformer-based architecture that converts
these features into a top-down representation
of the surrounding environment. The overall
architecture of BEVFormer is shown in bellow:

![bevformer](figs/bevformer_archi.png)

This repository is based on the official implementation of BEVFormer available here: https://github.com/fundamentalvision/BEVFormer.

Recall that in the context of our project, (i.e. building Tesla Autopilot system) we are only interested in extracing a bird-eye-view features map from surrounding cameras. From the figure above we therefore only want to extract BEV Bt that will be fed into downstream tasks (i.e. trajectories prediction and sim agents). However, to assess the quality of the BEV features map it is more convenient to tie it up with 3d detection tasks or segmentation tasks. Here, only 3d detection is used.

# Contribution
> **Temporal Enhanced Training of Multi-view 3D Object Detector via Historical
Object Prediction**
> - [Paper in arXiv](https://arxiv.org/abs/2304.00967), 3 Apr 2023

Very recent paper above comes up with a new
paradigm, named Historical Object Prediction
(HoP) for multi-view 3D detection to leverage
temporal information more effectively. The overall
architecture of HoP is shown in bellow: 

![hop](figs/hop_archi.png)

This method allows state-of-the-art architectures to perform even better by generating a pseudo BEV
feature map of timestamp (t−k) from its adjacent
frames and utilize this feature to predict the object
set at timestamp (t−k). HoP is performed only during
training and thus, does not introduce extra overheads
during inference. HoP is described as a plug-and-play approach and can be easily incorporated into
state-of-the-art BEV detection frameworks including BEVFormer.
As the paper is very recent, the implementation is
yet not available and therefore the main contribution
of this repo is the implementation of the proposed
HoP method including it to the BEVFormer model.

Our implementation of HoP can be found at `projects/mmdet3d_plugin/` and is as follow:

```
├── hop/
    ├── __init__.py
    ├── detectors/
        ├── __init__.py
        ├── bevformer_hop.py/
    ├── modules/
        ├── __init__.py
        ├── hop.py/
        ├── object_decoder.py/
        ├── temporal_decoder.py/
```

**HoP** framework is implemented as a `torch.nn.Module` in `hop.py`. The **TemporalEncoder** that combines the outputs of both **ShortTermTemporalDecoder** and **LongTermTemporalDecoder** is implemented in `temporal_decoder.py`. For the theoritical background behind the implementation, please refer to the original [paper](https://arxiv.org/pdf/2304.00967.pdf). We have chosen to reuse the object detection head (aka. DetectionTransformerDecoder) of the existing BEVFormer as the **ObjectDecoder** module in the **HoP** framework.

The **HoP** framework is then plugged to BEVFormer in `bevformer_hop.py` which implements the main class **BEVFormer_HoP**. Our contribution to the existing BEVFormer is the implementation of the forward pass using HoP, called `forward_hop()` and `forward_train()` as well as  custom initialization from existing pre-trained weights.
Then **BEVFormer_HoP** is register to the detectors registry to comply with the existing repo of BEVFormer implemented using [OpenMMLab](https://github.com/open-mmlab).


# Getting Started
- [Installation](docs/install.md) 
- [Dataset and Weights](docs/dataset.md)
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



