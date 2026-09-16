---
title: "ViewRefer: Grasp the Multi-view Knowledge for 3D Visual Grounding with GPT and Prototype Guidance"
entry_type: paper
source: "http://arxiv.org/pdf/2303.16894v2"
upstream_list: "awaisrauf/Awesome-CV-Foundational-Models"
year: 2023
authors: "Ziyu Guo,  Yiwen Tang,  Renrui Zhang,  Dong Wang,  Zhigang Wang,  Bin Zhao,  Xuelong Li"
code_url: "https://github.com/ZiyuGuo99/ViewRefer3D"
description: "Understanding 3D scenes from multi-view inputs has been proven to alleviate the view discrepancy issue in 3D visual grounding. However, existing methods normally neglect the view cues embedded in the text modality and fail to weigh the relative importance of different views. In this paper, we propose ViewRefer, a multi-view framework for 3D visual grounding exploring how to grasp the view knowledge from both text and 3D modalities. For the text branch, ViewRefer leverages the diverse linguistic knowledge of large-scale language models, e.g., GPT, to expand a single grounding text to multiple geometry-consistent descriptions. Meanwhile, in the 3D modality, a transformer fusion module with inter-view attention is introduced to boost the interaction of objects across views. On top of that, we further present a set of learnable multi-view prototypes, which memorize scene-agnostic knowledge for different views, and enhance the framework from two perspectives: a view-guided attention module for more robust text features, and a view-guided scoring strategy during the final prediction. With our designed paradigm, ViewRefer achieves superior performance on three benchmarks and surpasses the second-best by +2.8%, +1.2%, and +0.73% on Sr3D, Nr3D, and ScanRefer. Code will be released at https://github.com/ZiyuGuo99/ViewRefer3D."
---

# ViewRefer: Grasp the Multi-view Knowledge for 3D Visual Grounding with GPT and Prototype Guidance

**Source**: [http://arxiv.org/pdf/2303.16894v2](http://arxiv.org/pdf/2303.16894v2)

**Code**: [https://github.com/ZiyuGuo99/ViewRefer3D](https://github.com/ZiyuGuo99/ViewRefer3D)

**Year**: 2023

**Authors**: Ziyu Guo,  Yiwen Tang,  Renrui Zhang,  Dong Wang,  Zhigang Wang,  Bin Zhao,  Xuelong Li

## Description

Understanding 3D scenes from multi-view inputs has been proven to alleviate the view discrepancy issue in 3D visual grounding. However, existing methods normally neglect the view cues embedded in the text modality and fail to weigh the relative importance of different views. In this paper, we propose ViewRefer, a multi-view framework for 3D visual grounding exploring how to grasp the view knowledge from both text and 3D modalities. For the text branch, ViewRefer leverages the diverse linguistic knowledge of large-scale language models, e.g., GPT, to expand a single grounding text to multiple geometry-consistent descriptions. Meanwhile, in the 3D modality, a transformer fusion module with inter-view attention is introduced to boost the interaction of objects across views. On top of that, we further present a set of learnable multi-view prototypes, which memorize scene-agnostic knowledge for different views, and enhance the framework from two perspectives: a view-guided attention module for more robust text features, and a view-guided scoring strategy during the final prediction. With our designed paradigm, ViewRefer achieves superior performance on three benchmarks and surpasses the second-best by +2.8%, +1.2%, and +0.73% on Sr3D, Nr3D, and ScanRefer. Code will be released at https://github.com/ZiyuGuo99/ViewRefer3D.
