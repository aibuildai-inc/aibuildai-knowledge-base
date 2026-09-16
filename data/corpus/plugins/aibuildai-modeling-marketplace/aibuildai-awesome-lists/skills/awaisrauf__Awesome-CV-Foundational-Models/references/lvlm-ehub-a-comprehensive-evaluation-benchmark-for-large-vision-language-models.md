---
title: "LVLM-eHub: A Comprehensive Evaluation Benchmark for Large Vision-Language Models"
entry_type: paper
source: "http://arxiv.org/pdf/2306.09265v1"
upstream_list: "awaisrauf/Awesome-CV-Foundational-Models"
year: 2023
authors: "Peng Xu,  Wenqi Shao,  Kaipeng Zhang,  Peng Gao,  Shuo Liu,  Meng Lei,  Fanqing Meng,  Siyuan Huang,  Yu Qiao,  Ping Luo"
code_url: "https://github.com/OpenGVLab/Multi-Modality-Arena"
description: "Large Vision-Language Models (LVLMs) have recently played a dominant role in multimodal vision-language learning. Despite the great success, it lacks a holistic evaluation of their efficacy. This paper presents a comprehensive evaluation of publicly available large multimodal models by building a LVLM evaluation Hub (LVLM-eHub). Our LVLM-eHub consists of $8$ representative LVLMs such as InstructBLIP and MiniGPT-4, which are thoroughly evaluated by a quantitative capability evaluation and an online arena platform. The former evaluates $6$ categories of multimodal capabilities of LVLMs such as visual question answering and embodied artificial intelligence on $47$ standard text-related visual benchmarks, while the latter provides the user-level evaluation of LVLMs in an open-world question-answering scenario. The study reveals several innovative findings. First, instruction-tuned LVLM with massive in-domain data such as InstructBLIP heavily overfits many existing tasks, generalizing poorly in the open-world scenario. Second, instruction-tuned LVLM with moderate instruction-following data may result in object hallucination issues (i.e., generate objects that are inconsistent with target images in the descriptions). It either makes the current evaluation metric such as CIDEr for image captioning ineffective or generates wrong answers. Third, employing a multi-turn reasoning evaluation framework can mitigate the issue of object hallucination, shedding light on developing an effective pipeline for LVLM evaluation. The findings provide a foundational framework for the conception and assessment of innovative strategies aimed at enhancing zero-shot multimodal techniques. Our LVLM-eHub will be available at https://github.com/OpenGVLab/Multi-Modality-Arena"
---

# LVLM-eHub: A Comprehensive Evaluation Benchmark for Large Vision-Language Models

**Source**: [http://arxiv.org/pdf/2306.09265v1](http://arxiv.org/pdf/2306.09265v1)

**Code**: [https://github.com/OpenGVLab/Multi-Modality-Arena](https://github.com/OpenGVLab/Multi-Modality-Arena)

**Year**: 2023

**Authors**: Peng Xu,  Wenqi Shao,  Kaipeng Zhang,  Peng Gao,  Shuo Liu,  Meng Lei,  Fanqing Meng,  Siyuan Huang,  Yu Qiao,  Ping Luo

## Description

Large Vision-Language Models (LVLMs) have recently played a dominant role in multimodal vision-language learning. Despite the great success, it lacks a holistic evaluation of their efficacy. This paper presents a comprehensive evaluation of publicly available large multimodal models by building a LVLM evaluation Hub (LVLM-eHub). Our LVLM-eHub consists of $8$ representative LVLMs such as InstructBLIP and MiniGPT-4, which are thoroughly evaluated by a quantitative capability evaluation and an online arena platform. The former evaluates $6$ categories of multimodal capabilities of LVLMs such as visual question answering and embodied artificial intelligence on $47$ standard text-related visual benchmarks, while the latter provides the user-level evaluation of LVLMs in an open-world question-answering scenario. The study reveals several innovative findings. First, instruction-tuned LVLM with massive in-domain data such as InstructBLIP heavily overfits many existing tasks, generalizing poorly in the open-world scenario. Second, instruction-tuned LVLM with moderate instruction-following data may result in object hallucination issues (i.e., generate objects that are inconsistent with target images in the descriptions). It either makes the current evaluation metric such as CIDEr for image captioning ineffective or generates wrong answers. Third, employing a multi-turn reasoning evaluation framework can mitigate the issue of object hallucination, shedding light on developing an effective pipeline for LVLM evaluation. The findings provide a foundational framework for the conception and assessment of innovative strategies aimed at enhancing zero-shot multimodal techniques. Our LVLM-eHub will be available at https://github.com/OpenGVLab/Multi-Modality-Arena
