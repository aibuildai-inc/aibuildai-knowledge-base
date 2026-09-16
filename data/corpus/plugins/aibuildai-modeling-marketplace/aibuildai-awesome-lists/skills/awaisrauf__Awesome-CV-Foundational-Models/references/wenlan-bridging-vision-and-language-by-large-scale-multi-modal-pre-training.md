---
title: "WenLan: Bridging Vision and Language by Large-Scale Multi-Modal Pre-Training"
entry_type: paper
source: "http://arxiv.org/pdf/2103.06561v6"
upstream_list: "awaisrauf/Awesome-CV-Foundational-Models"
year: 2021
authors: "Huo, Yuqi,  Zhang, Manli,  Liu, Guangzhen,  Lu, Haoyu,  Gao, Yizhao,  Yang, Guoxing,  Wen, Jingyuan,  Zhang, Heng,  Xu, Baogui,  Zheng, Weihao,  others"
description: "Multi-modal pre-training models have been intensively explored to bridge vision and language in recent years. However, most of them explicitly model the cross-modal interaction between image-text pairs, by assuming that there exists strong semantic correlation between the text and image modalities. Since this strong assumption is often invalid in real-world scenarios, we choose to implicitly model the cross-modal correlation for large-scale multi-modal pre-training, which is the focus of the Chinese project `WenLan' led by our team. Specifically, with the weak correlation assumption over image-text pairs, we propose a two-tower pre-training model called BriVL within the cross-modal contrastive learning framework. Unlike OpenAI CLIP that adopts a simple contrastive learning method, we devise a more advanced algorithm by adapting the latest method MoCo into the cross-modal scenario. By building a large queue-based dictionary, our BriVL can incorporate more negative samples in limited GPU resources. We further construct a large Chinese multi-source image-text dataset called RUC-CAS-WenLan for pre-training our BriVL model. Extensive experiments demonstrate that the pre-trained BriVL model outperforms both UNITER and OpenAI CLIP on various downstream tasks."
---

# WenLan: Bridging Vision and Language by Large-Scale Multi-Modal Pre-Training

**Source**: [http://arxiv.org/pdf/2103.06561v6](http://arxiv.org/pdf/2103.06561v6)

**Year**: 2021

**Authors**: Huo, Yuqi,  Zhang, Manli,  Liu, Guangzhen,  Lu, Haoyu,  Gao, Yizhao,  Yang, Guoxing,  Wen, Jingyuan,  Zhang, Heng,  Xu, Baogui,  Zheng, Weihao,  others

## Description

Multi-modal pre-training models have been intensively explored to bridge vision and language in recent years. However, most of them explicitly model the cross-modal interaction between image-text pairs, by assuming that there exists strong semantic correlation between the text and image modalities. Since this strong assumption is often invalid in real-world scenarios, we choose to implicitly model the cross-modal correlation for large-scale multi-modal pre-training, which is the focus of the Chinese project `WenLan' led by our team. Specifically, with the weak correlation assumption over image-text pairs, we propose a two-tower pre-training model called BriVL within the cross-modal contrastive learning framework. Unlike OpenAI CLIP that adopts a simple contrastive learning method, we devise a more advanced algorithm by adapting the latest method MoCo into the cross-modal scenario. By building a large queue-based dictionary, our BriVL can incorporate more negative samples in limited GPU resources. We further construct a large Chinese multi-source image-text dataset called RUC-CAS-WenLan for pre-training our BriVL model. Extensive experiments demonstrate that the pre-trained BriVL model outperforms both UNITER and OpenAI CLIP on various downstream tasks.
