---
title: "A Unified Sequence Interface for Vision Tasks"
entry_type: paper
source: "http://arxiv.org/pdf/2206.07669v2"
upstream_list: "awaisrauf/Awesome-CV-Foundational-Models"
year: 2022
authors: "Chen, Ting,  Saxena, Saurabh,  Li, Lala,  Lin, Tsung-Yi,  Fleet, David J,  Hinton, Geoffrey E"
description: "While language tasks are naturally expressed in a single, unified, modeling framework, i.e., generating sequences of tokens, this has not been the case in computer vision. As a result, there is a proliferation of distinct architectures and loss functions for different vision tasks. In this work we show that a diverse set of \"core\" computer vision tasks can also be unified if formulated in terms of a shared pixel-to-sequence interface. We focus on four tasks, namely, object detection, instance segmentation, keypoint detection, and image captioning, all with diverse types of outputs, e.g., bounding boxes or dense masks. Despite that, by formulating the output of each task as a sequence of discrete tokens with a unified interface, we show that one can train a neural network with a single model architecture and loss function on all these tasks, with no task-specific customization. To solve a specific task, we use a short prompt as task description, and the sequence output adapts to the prompt so it can produce task-specific output. We show that such a model can achieve competitive performance compared to well-established task-specific models."
---

# A Unified Sequence Interface for Vision Tasks

**Source**: [http://arxiv.org/pdf/2206.07669v2](http://arxiv.org/pdf/2206.07669v2)

**Year**: 2022

**Authors**: Chen, Ting,  Saxena, Saurabh,  Li, Lala,  Lin, Tsung-Yi,  Fleet, David J,  Hinton, Geoffrey E

## Description

While language tasks are naturally expressed in a single, unified, modeling framework, i.e., generating sequences of tokens, this has not been the case in computer vision. As a result, there is a proliferation of distinct architectures and loss functions for different vision tasks. In this work we show that a diverse set of "core" computer vision tasks can also be unified if formulated in terms of a shared pixel-to-sequence interface. We focus on four tasks, namely, object detection, instance segmentation, keypoint detection, and image captioning, all with diverse types of outputs, e.g., bounding boxes or dense masks. Despite that, by formulating the output of each task as a sequence of discrete tokens with a unified interface, we show that one can train a neural network with a single model architecture and loss function on all these tasks, with no task-specific customization. To solve a specific task, we use a short prompt as task description, and the sequence output adapts to the prompt so it can produce task-specific output. We show that such a model can achieve competitive performance compared to well-established task-specific models.
