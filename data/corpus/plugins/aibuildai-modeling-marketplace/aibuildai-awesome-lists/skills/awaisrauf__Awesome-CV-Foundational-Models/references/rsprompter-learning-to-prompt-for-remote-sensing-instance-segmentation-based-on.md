---
title: "RSPrompter: Learning to Prompt for Remote Sensing Instance Segmentation based on Visual Foundation Model"
entry_type: paper
source: "http://arxiv.org/pdf/2306.16269v1"
upstream_list: "awaisrauf/Awesome-CV-Foundational-Models"
year: 2023
authors: "Keyan Chen,  Chenyang Liu,  Hao Chen,  Haotian Zhang,  Wenyuan Li,  Zhengxia Zou,  Zhenwei Shi"
description: "Leveraging vast training data (SA-1B), the foundation Segment Anything Model (SAM) proposed by Meta AI Research exhibits remarkable generalization and zero-shot capabilities. Nonetheless, as a category-agnostic instance segmentation method, SAM heavily depends on prior manual guidance involving points, boxes, and coarse-grained masks. Additionally, its performance on remote sensing image segmentation tasks has yet to be fully explored and demonstrated. In this paper, we consider designing an automated instance segmentation approach for remote sensing images based on the SAM foundation model, incorporating semantic category information. Inspired by prompt learning, we propose a method to learn the generation of appropriate prompts for SAM input. This enables SAM to produce semantically discernible segmentation results for remote sensing images, which we refer to as RSPrompter. We also suggest several ongoing derivatives for instance segmentation tasks, based on recent developments in the SAM community, and compare their performance with RSPrompter. Extensive experimental results on the WHU building, NWPU VHR-10, and SSDD datasets validate the efficacy of our proposed method. Our code is accessible at \\url{https://kyanchen.github.io/RSPrompter}."
---

# RSPrompter: Learning to Prompt for Remote Sensing Instance Segmentation based on Visual Foundation Model

**Source**: [http://arxiv.org/pdf/2306.16269v1](http://arxiv.org/pdf/2306.16269v1)

**Year**: 2023

**Authors**: Keyan Chen,  Chenyang Liu,  Hao Chen,  Haotian Zhang,  Wenyuan Li,  Zhengxia Zou,  Zhenwei Shi

## Description

Leveraging vast training data (SA-1B), the foundation Segment Anything Model (SAM) proposed by Meta AI Research exhibits remarkable generalization and zero-shot capabilities. Nonetheless, as a category-agnostic instance segmentation method, SAM heavily depends on prior manual guidance involving points, boxes, and coarse-grained masks. Additionally, its performance on remote sensing image segmentation tasks has yet to be fully explored and demonstrated. In this paper, we consider designing an automated instance segmentation approach for remote sensing images based on the SAM foundation model, incorporating semantic category information. Inspired by prompt learning, we propose a method to learn the generation of appropriate prompts for SAM input. This enables SAM to produce semantically discernible segmentation results for remote sensing images, which we refer to as RSPrompter. We also suggest several ongoing derivatives for instance segmentation tasks, based on recent developments in the SAM community, and compare their performance with RSPrompter. Extensive experimental results on the WHU building, NWPU VHR-10, and SSDD datasets validate the efficacy of our proposed method. Our code is accessible at \url{https://kyanchen.github.io/RSPrompter}.
