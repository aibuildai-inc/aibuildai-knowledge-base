---
title: "DeSAM: Decoupling Segment Anything Model for Generalizable Medical Image Segmentation"
entry_type: paper
source: "http://arxiv.org/pdf/2306.00499v1"
upstream_list: "awaisrauf/Awesome-CV-Foundational-Models"
year: 2023
authors: "Gao, Yifan,  Xia, Wei,  Hu, Dingdu,  Gao, Xin"
code_url: "https://github.com/yifangao112/DeSAM"
description: "Deep learning based automatic medical image segmentation models often suffer from domain shift, where the models trained on a source domain do not generalize well to other unseen domains. As a vision foundation model with powerful generalization capabilities, Segment Anything Model (SAM) shows potential for improving the cross-domain robustness of medical image segmentation. However, SAM and its fine-tuned models performed significantly worse in fully automatic mode compared to when given manual prompts. Upon further investigation, we discovered that the degradation in performance was related to the coupling effect of poor prompts and mask segmentation. In fully automatic mode, the presence of inevitable poor prompts (such as points outside the mask or boxes significantly larger than the mask) can significantly mislead mask generation. To address the coupling effect, we propose the decoupling SAM (DeSAM). DeSAM modifies SAM's mask decoder to decouple mask generation and prompt embeddings while leveraging pre-trained weights. We conducted experiments on publicly available prostate cross-site datasets. The results show that DeSAM improves dice score by an average of 8.96% (from 70.06% to 79.02%) compared to previous state-of-the-art domain generalization method. Moreover, DeSAM can be trained on personal devices with entry-level GPU since our approach does not rely on tuning the heavyweight image encoder. The code is publicly available at https://github.com/yifangao112/DeSAM."
---

# DeSAM: Decoupling Segment Anything Model for Generalizable Medical Image Segmentation

**Source**: [http://arxiv.org/pdf/2306.00499v1](http://arxiv.org/pdf/2306.00499v1)

**Code**: [https://github.com/yifangao112/DeSAM](https://github.com/yifangao112/DeSAM)

**Year**: 2023

**Authors**: Gao, Yifan,  Xia, Wei,  Hu, Dingdu,  Gao, Xin

## Description

Deep learning based automatic medical image segmentation models often suffer from domain shift, where the models trained on a source domain do not generalize well to other unseen domains. As a vision foundation model with powerful generalization capabilities, Segment Anything Model (SAM) shows potential for improving the cross-domain robustness of medical image segmentation. However, SAM and its fine-tuned models performed significantly worse in fully automatic mode compared to when given manual prompts. Upon further investigation, we discovered that the degradation in performance was related to the coupling effect of poor prompts and mask segmentation. In fully automatic mode, the presence of inevitable poor prompts (such as points outside the mask or boxes significantly larger than the mask) can significantly mislead mask generation. To address the coupling effect, we propose the decoupling SAM (DeSAM). DeSAM modifies SAM's mask decoder to decouple mask generation and prompt embeddings while leveraging pre-trained weights. We conducted experiments on publicly available prostate cross-site datasets. The results show that DeSAM improves dice score by an average of 8.96% (from 70.06% to 79.02%) compared to previous state-of-the-art domain generalization method. Moreover, DeSAM can be trained on personal devices with entry-level GPU since our approach does not rely on tuning the heavyweight image encoder. The code is publicly available at https://github.com/yifangao112/DeSAM.
