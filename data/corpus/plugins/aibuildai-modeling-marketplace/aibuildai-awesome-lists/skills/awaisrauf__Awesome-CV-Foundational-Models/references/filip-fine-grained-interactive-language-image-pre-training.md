---
title: "FILIP: Fine-grained Interactive Language-Image Pre-Training"
entry_type: paper
source: "http://arxiv.org/pdf/2111.07783v1"
upstream_list: "awaisrauf/Awesome-CV-Foundational-Models"
year: 2021
authors: "Yao, Lewei,  Huang, Runhui,  Hou, Lu,  Lu, Guansong,  Niu, Minzhe,  Xu, Hang,  Liang, Xiaodan,  Li, Zhenguo,  Jiang, Xin,  Xu, Chunjing"
description: "Unsupervised large-scale vision-language pre-training has shown promising advances on various downstream tasks. Existing methods often model the cross-modal interaction either via the similarity of the global feature of each modality which misses sufficient information, or finer-grained interactions using cross/self-attention upon visual and textual tokens. However, cross/self-attention suffers from inferior efficiency in both training and inference. In this paper, we introduce a large-scale Fine-grained Interactive Language-Image Pre-training (FILIP) to achieve finer-level alignment through a cross-modal late interaction mechanism, which uses a token-wise maximum similarity between visual and textual tokens to guide the contrastive objective. FILIP successfully leverages the finer-grained expressiveness between image patches and textual words by modifying only contrastive loss, while simultaneously gaining the ability to pre-compute image and text representations offline at inference, keeping both large-scale training and inference efficient. Furthermore, we construct a new large-scale image-text pair dataset called FILIP300M for pre-training. Experiments show that FILIP achieves state-of-the-art performance on multiple downstream vision-language tasks including zero-shot image classification and image-text retrieval. The visualization on word-patch alignment further shows that FILIP can learn meaningful fine-grained features with promising localization ability."
---

# FILIP: Fine-grained Interactive Language-Image Pre-Training

**Source**: [http://arxiv.org/pdf/2111.07783v1](http://arxiv.org/pdf/2111.07783v1)

**Year**: 2021

**Authors**: Yao, Lewei,  Huang, Runhui,  Hou, Lu,  Lu, Guansong,  Niu, Minzhe,  Xu, Hang,  Liang, Xiaodan,  Li, Zhenguo,  Jiang, Xin,  Xu, Chunjing

## Description

Unsupervised large-scale vision-language pre-training has shown promising advances on various downstream tasks. Existing methods often model the cross-modal interaction either via the similarity of the global feature of each modality which misses sufficient information, or finer-grained interactions using cross/self-attention upon visual and textual tokens. However, cross/self-attention suffers from inferior efficiency in both training and inference. In this paper, we introduce a large-scale Fine-grained Interactive Language-Image Pre-training (FILIP) to achieve finer-level alignment through a cross-modal late interaction mechanism, which uses a token-wise maximum similarity between visual and textual tokens to guide the contrastive objective. FILIP successfully leverages the finer-grained expressiveness between image patches and textual words by modifying only contrastive loss, while simultaneously gaining the ability to pre-compute image and text representations offline at inference, keeping both large-scale training and inference efficient. Furthermore, we construct a new large-scale image-text pair dataset called FILIP300M for pre-training. Experiments show that FILIP achieves state-of-the-art performance on multiple downstream vision-language tasks including zero-shot image classification and image-text retrieval. The visualization on word-patch alignment further shows that FILIP can learn meaningful fine-grained features with promising localization ability.
