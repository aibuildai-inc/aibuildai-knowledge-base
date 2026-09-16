---
title: "GIT: A Generative Image-to-text Transformer for Vision and Language"
entry_type: paper
source: "http://arxiv.org/pdf/2205.14100v5"
upstream_list: "awaisrauf/Awesome-CV-Foundational-Models"
year: 2022
authors: "Wang, Jianfeng,  Yang, Zhengyuan,  Hu, Xiaowei,  Li, Linjie,  Lin, Kevin,  Gan, Zhe,  Liu, Zicheng,  Liu, Ce,  Wang, Lijuan"
code_url: "https://github.com/microsoft/GenerativeImage2Text"
description: "In this paper, we design and train a Generative Image-to-text Transformer, GIT, to unify vision-language tasks such as image/video captioning and question answering. While generative models provide a consistent network architecture between pre-training and fine-tuning, existing work typically contains complex structures (uni/multi-modal encoder/decoder) and depends on external modules such as object detectors/taggers and optical character recognition (OCR). In GIT, we simplify the architecture as one image encoder and one text decoder under a single language modeling task. We also scale up the pre-training data and the model size to boost the model performance. Without bells and whistles, our GIT establishes new state of the arts on 12 challenging benchmarks with a large margin. For instance, our model surpasses the human performance for the first time on TextCaps (138.2 vs. 125.5 in CIDEr). Furthermore, we present a new scheme of generation-based image classification and scene text recognition, achieving decent performance on standard benchmarks. Codes are released at \\url{https://github.com/microsoft/GenerativeImage2Text}."
---

# GIT: A Generative Image-to-text Transformer for Vision and Language

**Source**: [http://arxiv.org/pdf/2205.14100v5](http://arxiv.org/pdf/2205.14100v5)

**Code**: [https://github.com/microsoft/GenerativeImage2Text](https://github.com/microsoft/GenerativeImage2Text)

**Year**: 2022

**Authors**: Wang, Jianfeng,  Yang, Zhengyuan,  Hu, Xiaowei,  Li, Linjie,  Lin, Kevin,  Gan, Zhe,  Liu, Zicheng,  Liu, Ce,  Wang, Lijuan

## Description

In this paper, we design and train a Generative Image-to-text Transformer, GIT, to unify vision-language tasks such as image/video captioning and question answering. While generative models provide a consistent network architecture between pre-training and fine-tuning, existing work typically contains complex structures (uni/multi-modal encoder/decoder) and depends on external modules such as object detectors/taggers and optical character recognition (OCR). In GIT, we simplify the architecture as one image encoder and one text decoder under a single language modeling task. We also scale up the pre-training data and the model size to boost the model performance. Without bells and whistles, our GIT establishes new state of the arts on 12 challenging benchmarks with a large margin. For instance, our model surpasses the human performance for the first time on TextCaps (138.2 vs. 125.5 in CIDEr). Furthermore, we present a new scheme of generation-based image classification and scene text recognition, achieving decent performance on standard benchmarks. Codes are released at \url{https://github.com/microsoft/GenerativeImage2Text}.
