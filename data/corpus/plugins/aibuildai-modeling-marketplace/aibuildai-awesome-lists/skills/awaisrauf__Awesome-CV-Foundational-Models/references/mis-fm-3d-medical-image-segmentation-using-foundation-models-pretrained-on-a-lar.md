---
title: "MIS-FM: 3D Medical Image Segmentation using Foundation Models Pretrained on a Large-Scale Unannotated Dataset"
entry_type: paper
source: "http://arxiv.org/pdf/2306.16925v1"
upstream_list: "awaisrauf/Awesome-CV-Foundational-Models"
year: 2023
authors: "Guotai Wang, Jianghao Wu, Xiangde Luo, Xinglong Liu, Kang Li, Shaoting Zhang"
code_url: "https://github.com/openmedlab/MIS-FM"
description: "Pretraining with large-scale 3D volumes has a potential for improving the segmentation performance on a target medical image dataset where the training images and annotations are limited. Due to the high cost of acquiring pixel-level segmentation annotations on the large-scale pretraining dataset, pretraining with unannotated images is highly desirable. In this work, we propose a novel self-supervised learning strategy named Volume Fusion (VF) for pretraining 3D segmentation models. It fuses several random patches from a foreground sub-volume to a background sub-volume based on a predefined set of discrete fusion coefficients, and forces the model to predict the fusion coefficient of each voxel, which is formulated as a self-supervised segmentation task without manual annotations. Additionally, we propose a novel network architecture based on parallel convolution and transformer blocks that is suitable to be transferred to different downstream segmentation tasks with various scales of organs and lesions. The proposed model was pretrained with 110k unannotated 3D CT volumes, and experiments with different downstream segmentation targets including head and neck organs, thoracic/abdominal organs showed that our pretrained model largely outperformed training from scratch and several state-of-the-art self-supervised training methods and segmentation models. The code and pretrained model are available at https://github.com/openmedlab/MIS-FM."
---

# MIS-FM: 3D Medical Image Segmentation using Foundation Models Pretrained on a Large-Scale Unannotated Dataset

**Source**: [http://arxiv.org/pdf/2306.16925v1](http://arxiv.org/pdf/2306.16925v1)

**Code**: [https://github.com/openmedlab/MIS-FM](https://github.com/openmedlab/MIS-FM)

**Year**: 2023

**Authors**: Guotai Wang, Jianghao Wu, Xiangde Luo, Xinglong Liu, Kang Li, Shaoting Zhang

## Description

Pretraining with large-scale 3D volumes has a potential for improving the segmentation performance on a target medical image dataset where the training images and annotations are limited. Due to the high cost of acquiring pixel-level segmentation annotations on the large-scale pretraining dataset, pretraining with unannotated images is highly desirable. In this work, we propose a novel self-supervised learning strategy named Volume Fusion (VF) for pretraining 3D segmentation models. It fuses several random patches from a foreground sub-volume to a background sub-volume based on a predefined set of discrete fusion coefficients, and forces the model to predict the fusion coefficient of each voxel, which is formulated as a self-supervised segmentation task without manual annotations. Additionally, we propose a novel network architecture based on parallel convolution and transformer blocks that is suitable to be transferred to different downstream segmentation tasks with various scales of organs and lesions. The proposed model was pretrained with 110k unannotated 3D CT volumes, and experiments with different downstream segmentation targets including head and neck organs, thoracic/abdominal organs showed that our pretrained model largely outperformed training from scratch and several state-of-the-art self-supervised training methods and segmentation models. The code and pretrained model are available at https://github.com/openmedlab/MIS-FM.
