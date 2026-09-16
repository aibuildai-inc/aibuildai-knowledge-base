---
title: "TaCA: Upgrading Your Visual Foundation Model with Task-agnostic Compatible Adapter"
entry_type: paper
source: "http://arxiv.org/pdf/2306.12642v1"
upstream_list: "awaisrauf/Awesome-CV-Foundational-Models"
year: 2023
authors: "Binjie Zhang,  Yixiao Ge,  Xuyuan Xu,  Ying Shan,  Mike Zheng Shou"
code_url: "https://github.com/TencentARC/TaCA"
description: "Visual foundation models like CLIP excel in learning feature representations from extensive datasets through self-supervised methods, demonstrating remarkable transfer learning and generalization capabilities. A growing number of applications based on visual foundation models are emerging, including innovative solutions such as BLIP-2. These applications employ pre-trained CLIP models as upstream feature extractors and train various downstream modules to accomplish diverse tasks. In situations involving system upgrades that require updating the upstream foundation model, it becomes essential to re-train all downstream modules to adapt to the new foundation model, which is inflexible and inefficient. In this paper, we introduce a parameter-efficient and task-agnostic adapter, dubbed TaCA, that facilitates compatibility across distinct foundation models while ensuring enhanced performance for the new models. TaCA allows downstream applications to seamlessly integrate better-performing foundation models without necessitating retraining. We conduct extensive experimental validation of TaCA using different scales of models with up to one billion parameters on various tasks such as video-text retrieval, video recognition, and visual question answering. The results consistently demonstrate the emergent ability of TaCA on hot-plugging upgrades for visual foundation models. Codes and models will be available at https://github.com/TencentARC/TaCA."
---

# TaCA: Upgrading Your Visual Foundation Model with Task-agnostic Compatible Adapter

**Source**: [http://arxiv.org/pdf/2306.12642v1](http://arxiv.org/pdf/2306.12642v1)

**Code**: [https://github.com/TencentARC/TaCA](https://github.com/TencentARC/TaCA)

**Year**: 2023

**Authors**: Binjie Zhang,  Yixiao Ge,  Xuyuan Xu,  Ying Shan,  Mike Zheng Shou

## Description

Visual foundation models like CLIP excel in learning feature representations from extensive datasets through self-supervised methods, demonstrating remarkable transfer learning and generalization capabilities. A growing number of applications based on visual foundation models are emerging, including innovative solutions such as BLIP-2. These applications employ pre-trained CLIP models as upstream feature extractors and train various downstream modules to accomplish diverse tasks. In situations involving system upgrades that require updating the upstream foundation model, it becomes essential to re-train all downstream modules to adapt to the new foundation model, which is inflexible and inefficient. In this paper, we introduce a parameter-efficient and task-agnostic adapter, dubbed TaCA, that facilitates compatibility across distinct foundation models while ensuring enhanced performance for the new models. TaCA allows downstream applications to seamlessly integrate better-performing foundation models without necessitating retraining. We conduct extensive experimental validation of TaCA using different scales of models with up to one billion parameters on various tasks such as video-text retrieval, video recognition, and visual question answering. The results consistently demonstrate the emergent ability of TaCA on hot-plugging upgrades for visual foundation models. Codes and models will be available at https://github.com/TencentARC/TaCA.
