---
title: "Simple Open-Vocabulary Object Detection with Vision Transformers"
entry_type: paper
source: "http://arxiv.org/pdf/2205.06230v2"
upstream_list: "awaisrauf/Awesome-CV-Foundational-Models"
year: 2022
authors: "Matthias Minderer,  Alexey Gritsenko,  Austin Stone,  Maxim Neumann,  Dirk Weissenborn,  Alexey Dosovitskiy,  Aravindh Mahendran,  Anurag Arnab,  Mostafa Dehghani,  Zhuoran Shen,  Xiao Wang,  Xiaohua Zhai,  Thomas Kipf,  Neil Houlsby"
description: "Combining simple architectures with large-scale pre-training has led to massive improvements in image classification. For object detection, pre-training and scaling approaches are less well established, especially in the long-tailed and open-vocabulary setting, where training data is relatively scarce. In this paper, we propose a strong recipe for transferring image-text models to open-vocabulary object detection. We use a standard Vision Transformer architecture with minimal modifications, contrastive image-text pre-training, and end-to-end detection fine-tuning. Our analysis of the scaling properties of this setup shows that increasing image-level pre-training and model size yield consistent improvements on the downstream detection task. We provide the adaptation strategies and regularizations needed to attain very strong performance on zero-shot text-conditioned and one-shot image-conditioned object detection. Code and models are available on GitHub."
---

# Simple Open-Vocabulary Object Detection with Vision Transformers

**Source**: [http://arxiv.org/pdf/2205.06230v2](http://arxiv.org/pdf/2205.06230v2)

**Year**: 2022

**Authors**: Matthias Minderer,  Alexey Gritsenko,  Austin Stone,  Maxim Neumann,  Dirk Weissenborn,  Alexey Dosovitskiy,  Aravindh Mahendran,  Anurag Arnab,  Mostafa Dehghani,  Zhuoran Shen,  Xiao Wang,  Xiaohua Zhai,  Thomas Kipf,  Neil Houlsby

## Description

Combining simple architectures with large-scale pre-training has led to massive improvements in image classification. For object detection, pre-training and scaling approaches are less well established, especially in the long-tailed and open-vocabulary setting, where training data is relatively scarce. In this paper, we propose a strong recipe for transferring image-text models to open-vocabulary object detection. We use a standard Vision Transformer architecture with minimal modifications, contrastive image-text pre-training, and end-to-end detection fine-tuning. Our analysis of the scaling properties of this setup shows that increasing image-level pre-training and model size yield consistent improvements on the downstream detection task. We provide the adaptation strategies and regularizations needed to attain very strong performance on zero-shot text-conditioned and one-shot image-conditioned object detection. Code and models are available on GitHub.
