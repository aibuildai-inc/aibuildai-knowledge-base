---
title: "InstructBLIP: Towards General-purpose Vision-Language Models with Instruction Tuning"
entry_type: paper
source: "http://arxiv.org/pdf/2305.06500v2"
upstream_list: "awaisrauf/Awesome-CV-Foundational-Models"
year: 2023
authors: "Wenliang Dai,  Junnan Li,  Dongxu Li,  Anthony Meng Huat Tiong,  Junqi Zhao,  Weisheng Wang,  Boyang Li,  Pascale Fung,  Steven Hoi"
code_url: "https://github.com/salesforce/LAVIS"
description: "Large-scale pre-training and instruction tuning have been successful at creating general-purpose language models with broad competence. However, building general-purpose vision-language models is challenging due to the rich input distributions and task diversity resulting from the additional visual input. Although vision-language pretraining has been widely studied, vision-language instruction tuning remains under-explored. In this paper, we conduct a systematic and comprehensive study on vision-language instruction tuning based on the pretrained BLIP-2 models. We gather 26 publicly available datasets, covering a wide variety of tasks and capabilities, and transform them into instruction tuning format. Additionally, we introduce an instruction-aware Query Transformer, which extracts informative features tailored to the given instruction. Trained on 13 held-in datasets, InstructBLIP attains state-of-the-art zero-shot performance across all 13 held-out datasets, substantially outperforming BLIP-2 and larger Flamingo models. Our models also lead to state-of-the-art performance when finetuned on individual downstream tasks (e.g., 90.7% accuracy on ScienceQA questions with image contexts). Furthermore, we qualitatively demonstrate the advantages of InstructBLIP over concurrent multimodal models. All InstructBLIP models are open-sourced at https://github.com/salesforce/LAVIS/tree/main/projects/instructblip."
---

# InstructBLIP: Towards General-purpose Vision-Language Models with Instruction Tuning

**Source**: [http://arxiv.org/pdf/2305.06500v2](http://arxiv.org/pdf/2305.06500v2)

**Code**: [https://github.com/salesforce/LAVIS](https://github.com/salesforce/LAVIS)

**Year**: 2023

**Authors**: Wenliang Dai,  Junnan Li,  Dongxu Li,  Anthony Meng Huat Tiong,  Junqi Zhao,  Weisheng Wang,  Boyang Li,  Pascale Fung,  Steven Hoi

## Description

Large-scale pre-training and instruction tuning have been successful at creating general-purpose language models with broad competence. However, building general-purpose vision-language models is challenging due to the rich input distributions and task diversity resulting from the additional visual input. Although vision-language pretraining has been widely studied, vision-language instruction tuning remains under-explored. In this paper, we conduct a systematic and comprehensive study on vision-language instruction tuning based on the pretrained BLIP-2 models. We gather 26 publicly available datasets, covering a wide variety of tasks and capabilities, and transform them into instruction tuning format. Additionally, we introduce an instruction-aware Query Transformer, which extracts informative features tailored to the given instruction. Trained on 13 held-in datasets, InstructBLIP attains state-of-the-art zero-shot performance across all 13 held-out datasets, substantially outperforming BLIP-2 and larger Flamingo models. Our models also lead to state-of-the-art performance when finetuned on individual downstream tasks (e.g., 90.7% accuracy on ScienceQA questions with image contexts). Furthermore, we qualitatively demonstrate the advantages of InstructBLIP over concurrent multimodal models. All InstructBLIP models are open-sourced at https://github.com/salesforce/LAVIS/tree/main/projects/instructblip.
