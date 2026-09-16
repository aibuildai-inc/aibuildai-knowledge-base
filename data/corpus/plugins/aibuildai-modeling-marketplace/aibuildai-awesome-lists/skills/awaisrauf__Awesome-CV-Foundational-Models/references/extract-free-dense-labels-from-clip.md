---
title: "Extract Free Dense Labels from CLIP"
entry_type: paper
source: "http://arxiv.org/pdf/2112.01071v2"
upstream_list: "awaisrauf/Awesome-CV-Foundational-Models"
year: 2021
authors: "Zhou, Chong,  Loy, Chen Change,  Dai, Bo"
code_url: "https://github.com/chongzhou96/MaskCLIP"
description: "Contrastive Language-Image Pre-training (CLIP) has made a remarkable breakthrough in open-vocabulary zero-shot image recognition. Many recent studies leverage the pre-trained CLIP models for image-level classification and manipulation. In this paper, we wish examine the intrinsic potential of CLIP for pixel-level dense prediction, specifically in semantic segmentation. To this end, with minimal modification, we show that MaskCLIP yields compelling segmentation results on open concepts across various datasets in the absence of annotations and fine-tuning. By adding pseudo labeling and self-training, MaskCLIP+ surpasses SOTA transductive zero-shot semantic segmentation methods by large margins, e.g., mIoUs of unseen classes on PASCAL VOC/PASCAL Context/COCO Stuff are improved from 35.6/20.7/30.3 to 86.1/66.7/54.7. We also test the robustness of MaskCLIP under input corruption and evaluate its capability in discriminating fine-grained objects and novel concepts. Our finding suggests that MaskCLIP can serve as a new reliable source of supervision for dense prediction tasks to achieve annotation-free segmentation. Source code is available at https://github.com/chongzhou96/MaskCLIP."
---

# Extract Free Dense Labels from CLIP

**Source**: [http://arxiv.org/pdf/2112.01071v2](http://arxiv.org/pdf/2112.01071v2)

**Code**: [https://github.com/chongzhou96/MaskCLIP](https://github.com/chongzhou96/MaskCLIP)

**Year**: 2021

**Authors**: Zhou, Chong,  Loy, Chen Change,  Dai, Bo

## Description

Contrastive Language-Image Pre-training (CLIP) has made a remarkable breakthrough in open-vocabulary zero-shot image recognition. Many recent studies leverage the pre-trained CLIP models for image-level classification and manipulation. In this paper, we wish examine the intrinsic potential of CLIP for pixel-level dense prediction, specifically in semantic segmentation. To this end, with minimal modification, we show that MaskCLIP yields compelling segmentation results on open concepts across various datasets in the absence of annotations and fine-tuning. By adding pseudo labeling and self-training, MaskCLIP+ surpasses SOTA transductive zero-shot semantic segmentation methods by large margins, e.g., mIoUs of unseen classes on PASCAL VOC/PASCAL Context/COCO Stuff are improved from 35.6/20.7/30.3 to 86.1/66.7/54.7. We also test the robustness of MaskCLIP under input corruption and evaluate its capability in discriminating fine-grained objects and novel concepts. Our finding suggests that MaskCLIP can serve as a new reliable source of supervision for dense prediction tasks to achieve annotation-free segmentation. Source code is available at https://github.com/chongzhou96/MaskCLIP.
