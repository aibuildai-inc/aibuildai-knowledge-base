---
title: "Scaling Open-Vocabulary Image Segmentation with Image-Level Labels"
entry_type: paper
source: "http://arxiv.org/pdf/2112.12143v2"
upstream_list: "awaisrauf/Awesome-CV-Foundational-Models"
year: 2021
authors: "Golnaz Ghiasi,  Xiuye Gu,  Yin Cui,  Tsung-Yi Lin"
description: "We design an open-vocabulary image segmentation model to organize an image into meaningful regions indicated by arbitrary texts. Recent works (CLIP and ALIGN), despite attaining impressive open-vocabulary classification accuracy with image-level caption labels, are unable to segment visual concepts with pixels. We argue that these models miss an important step of visual grouping, which organizes pixels into groups before learning visual-semantic alignments. We propose OpenSeg to address the above issue while still making use of scalable image-level supervision of captions. First, it learns to propose segmentation masks for possible organizations. Then it learns visual-semantic alignments by aligning each word in a caption to one or a few predicted masks. We find the mask representations are the key to support learning image segmentation from captions, making it possible to scale up the dataset and vocabulary sizes. OpenSeg significantly outperforms the recent open-vocabulary method of LSeg by +19.9 mIoU on PASCAL dataset, thanks to its scalability."
---

# Scaling Open-Vocabulary Image Segmentation with Image-Level Labels

**Source**: [http://arxiv.org/pdf/2112.12143v2](http://arxiv.org/pdf/2112.12143v2)

**Year**: 2021

**Authors**: Golnaz Ghiasi,  Xiuye Gu,  Yin Cui,  Tsung-Yi Lin

## Description

We design an open-vocabulary image segmentation model to organize an image into meaningful regions indicated by arbitrary texts. Recent works (CLIP and ALIGN), despite attaining impressive open-vocabulary classification accuracy with image-level caption labels, are unable to segment visual concepts with pixels. We argue that these models miss an important step of visual grouping, which organizes pixels into groups before learning visual-semantic alignments. We propose OpenSeg to address the above issue while still making use of scalable image-level supervision of captions. First, it learns to propose segmentation masks for possible organizations. Then it learns visual-semantic alignments by aligning each word in a caption to one or a few predicted masks. We find the mask representations are the key to support learning image segmentation from captions, making it possible to scale up the dataset and vocabulary sizes. OpenSeg significantly outperforms the recent open-vocabulary method of LSeg by +19.9 mIoU on PASCAL dataset, thanks to its scalability.
