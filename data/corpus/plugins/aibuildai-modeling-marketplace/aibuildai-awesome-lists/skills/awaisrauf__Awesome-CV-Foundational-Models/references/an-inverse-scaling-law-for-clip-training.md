---
title: "An Inverse Scaling Law for CLIP Training"
entry_type: paper
source: "http://arxiv.org/pdf/2305.07017v1"
upstream_list: "awaisrauf/Awesome-CV-Foundational-Models"
year: 2023
authors: "Xianhang Li,  Zeyu Wang,  Cihang Xie"
code_url: "https://github.com/UCSC-VLAA/CLIPA"
description: "CLIP, the first foundation model that connects images and text, has enabled many recent breakthroughs in computer vision. However, its associated training cost is prohibitively high, imposing a significant barrier to its widespread exploration. In this paper, we present a surprising finding that there exists an inverse scaling law for CLIP training, whereby the larger the image/text encoders used, the shorter the sequence length of image/text tokens that can be applied in training. Moreover, we showcase that the strategy for reducing image/text token length plays a crucial role in determining the quality of this scaling law. As a result of this finding, we are able to successfully train CLIP even by using academic resources. For example, on an A100 eight-GPU server, our CLIP models achieve zero-shot top-1 ImageNet accuracies of 63.2% in ~2 days, 67.8% in ~3 days, and 69.3% in ~4 days. By reducing the computation barrier associated with CLIP, we hope to inspire more research in this field, particularly from academics. Our code is available at https://github.com/UCSC-VLAA/CLIPA."
---

# An Inverse Scaling Law for CLIP Training

**Source**: [http://arxiv.org/pdf/2305.07017v1](http://arxiv.org/pdf/2305.07017v1)

**Code**: [https://github.com/UCSC-VLAA/CLIPA](https://github.com/UCSC-VLAA/CLIPA)

**Year**: 2023

**Authors**: Xianhang Li,  Zeyu Wang,  Cihang Xie

## Description

CLIP, the first foundation model that connects images and text, has enabled many recent breakthroughs in computer vision. However, its associated training cost is prohibitively high, imposing a significant barrier to its widespread exploration. In this paper, we present a surprising finding that there exists an inverse scaling law for CLIP training, whereby the larger the image/text encoders used, the shorter the sequence length of image/text tokens that can be applied in training. Moreover, we showcase that the strategy for reducing image/text token length plays a crucial role in determining the quality of this scaling law. As a result of this finding, we are able to successfully train CLIP even by using academic resources. For example, on an A100 eight-GPU server, our CLIP models achieve zero-shot top-1 ImageNet accuracies of 63.2% in ~2 days, 67.8% in ~3 days, and 69.3% in ~4 days. By reducing the computation barrier associated with CLIP, we hope to inspire more research in this field, particularly from academics. Our code is available at https://github.com/UCSC-VLAA/CLIPA.
