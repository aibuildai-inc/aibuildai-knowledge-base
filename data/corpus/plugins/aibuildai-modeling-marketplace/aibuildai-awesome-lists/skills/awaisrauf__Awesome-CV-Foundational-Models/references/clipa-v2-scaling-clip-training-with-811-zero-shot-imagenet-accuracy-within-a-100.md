---
title: "CLIPA-v2: Scaling CLIP Training with 81.1% Zero-shot ImageNet Accuracy within a \\$10,000 Budget; An Extra \\$4,000 Unlocks 81.8% Accuracy"
entry_type: paper
source: "http://arxiv.org/pdf/2306.15658v1"
upstream_list: "awaisrauf/Awesome-CV-Foundational-Models"
year: 2023
authors: "Xianhang Li,  Zeyu Wang,  Cihang Xie"
code_url: "https://github.com/UCSC-VLAA/CLIPA"
description: "The recent work CLIPA presents an inverse scaling law for CLIP training -- whereby the larger the image/text encoders used, the shorter the sequence length of image/text tokens that can be applied in training. This finding enables us to train high-performance CLIP models with significantly reduced computations. Building upon this work, we hereby present CLIPA-v2 with two key contributions. Technically, we find this inverse scaling law is also applicable in the finetuning stage, enabling further reduction in computational needs. Empirically, we explore CLIPA at scale, extending the experiments up to the H/14 model with ~13B image-text pairs seen during training. Our results are exciting -- by only allocating a budget of \\$10,000, our CLIP model achieves an impressive zero-shot ImageNet accuracy of 81.1%, surpassing the prior best CLIP model (from OpenCLIP, 80.1%) by 1.0% and meanwhile reducing the computational cost by ~39X. Moreover, with an additional investment of $4,000, we can further elevate the zero-shot ImageNet accuracy to 81.8%. Our code and models are available at https://github.com/UCSC-VLAA/CLIPA."
---

# CLIPA-v2: Scaling CLIP Training with 81.1% Zero-shot ImageNet Accuracy within a \$10,000 Budget; An Extra \$4,000 Unlocks 81.8% Accuracy

**Source**: [http://arxiv.org/pdf/2306.15658v1](http://arxiv.org/pdf/2306.15658v1)

**Code**: [https://github.com/UCSC-VLAA/CLIPA](https://github.com/UCSC-VLAA/CLIPA)

**Year**: 2023

**Authors**: Xianhang Li,  Zeyu Wang,  Cihang Xie

## Description

The recent work CLIPA presents an inverse scaling law for CLIP training -- whereby the larger the image/text encoders used, the shorter the sequence length of image/text tokens that can be applied in training. This finding enables us to train high-performance CLIP models with significantly reduced computations. Building upon this work, we hereby present CLIPA-v2 with two key contributions. Technically, we find this inverse scaling law is also applicable in the finetuning stage, enabling further reduction in computational needs. Empirically, we explore CLIPA at scale, extending the experiments up to the H/14 model with ~13B image-text pairs seen during training. Our results are exciting -- by only allocating a budget of \$10,000, our CLIP model achieves an impressive zero-shot ImageNet accuracy of 81.1%, surpassing the prior best CLIP model (from OpenCLIP, 80.1%) by 1.0% and meanwhile reducing the computational cost by ~39X. Moreover, with an additional investment of $4,000, we can further elevate the zero-shot ImageNet accuracy to 81.8%. Our code and models are available at https://github.com/UCSC-VLAA/CLIPA.
