---
title: "Unmasked Teacher: Towards Training-Efficient Video Foundation Models"
entry_type: paper
source: "http://arxiv.org/pdf/2303.16058v1"
upstream_list: "awaisrauf/Awesome-CV-Foundational-Models"
year: 2023
authors: "Li, Kunchang,  Wang, Yali,  Li, Yizhuo,  Wang, Yi,  He, Yinan,  Wang, Limin,  Qiao, Yu"
code_url: "https://github.com/OpenGVLab/unmasked_teacher"
description: "Video Foundation Models (VFMs) have received limited exploration due to high computational costs and data scarcity. Previous VFMs rely on Image Foundation Models (IFMs), which face challenges in transferring to the video domain. Although VideoMAE has trained a robust ViT from limited data, its low-level reconstruction poses convergence difficulties and conflicts with high-level cross-modal alignment. This paper proposes a training-efficient method for temporal-sensitive VFMs that integrates the benefits of existing methods. To increase data efficiency, we mask out most of the low-semantics video tokens, but selectively align the unmasked tokens with IFM, which serves as the UnMasked Teacher (UMT). By providing semantic guidance, our method enables faster convergence and multimodal friendliness. With a progressive pre-training framework, our model can handle various tasks including scene-related, temporal-related, and complex video-language understanding. Using only public sources for pre-training in 6 days on 32 A100 GPUs, our scratch-built ViT-L/16 achieves state-of-the-art performances on various video tasks. The code and models will be released at https://github.com/OpenGVLab/unmasked_teacher."
---

# Unmasked Teacher: Towards Training-Efficient Video Foundation Models

**Source**: [http://arxiv.org/pdf/2303.16058v1](http://arxiv.org/pdf/2303.16058v1)

**Code**: [https://github.com/OpenGVLab/unmasked_teacher](https://github.com/OpenGVLab/unmasked_teacher)

**Year**: 2023

**Authors**: Li, Kunchang,  Wang, Yali,  Li, Yizhuo,  Wang, Yi,  He, Yinan,  Wang, Limin,  Qiao, Yu

## Description

Video Foundation Models (VFMs) have received limited exploration due to high computational costs and data scarcity. Previous VFMs rely on Image Foundation Models (IFMs), which face challenges in transferring to the video domain. Although VideoMAE has trained a robust ViT from limited data, its low-level reconstruction poses convergence difficulties and conflicts with high-level cross-modal alignment. This paper proposes a training-efficient method for temporal-sensitive VFMs that integrates the benefits of existing methods. To increase data efficiency, we mask out most of the low-semantics video tokens, but selectively align the unmasked tokens with IFM, which serves as the UnMasked Teacher (UMT). By providing semantic guidance, our method enables faster convergence and multimodal friendliness. With a progressive pre-training framework, our model can handle various tasks including scene-related, temporal-related, and complex video-language understanding. Using only public sources for pre-training in 6 days on 32 A100 GPUs, our scratch-built ViT-L/16 achieves state-of-the-art performances on various video tasks. The code and models will be released at https://github.com/OpenGVLab/unmasked_teacher.
