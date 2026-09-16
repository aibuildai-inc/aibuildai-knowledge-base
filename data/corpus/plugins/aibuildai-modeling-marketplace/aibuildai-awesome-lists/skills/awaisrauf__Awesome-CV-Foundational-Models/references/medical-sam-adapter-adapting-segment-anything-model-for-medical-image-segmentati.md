---
title: "Medical SAM Adapter: Adapting Segment Anything Model for Medical Image Segmentation"
entry_type: paper
source: "http://arxiv.org/pdf/2304.12620v6"
upstream_list: "awaisrauf/Awesome-CV-Foundational-Models"
year: 2023
authors: "Wu, Junde,  Fu, Rao,  Fang, Huihui,  Liu, Yuanpei,  Wang, Zhaowei,  Xu, Yanwu,  Jin, Yueming,  Arbel, Tal"
code_url: "https://github.com/WuJunde/Medical-SAM-Adapter"
description: "The Segment Anything Model (SAM) has recently gained popularity in the field of image segmentation. Thanks to its impressive capabilities in all-round segmentation tasks and its prompt-based interface, SAM has sparked intensive discussion within the community. It is even said by many prestigious experts that image segmentation task has been \"finished\" by SAM. However, medical image segmentation, although an important branch of the image segmentation family, seems not to be included in the scope of Segmenting \"Anything\". Many individual experiments and recent studies have shown that SAM performs subpar in medical image segmentation. A natural question is how to find the missing piece of the puzzle to extend the strong segmentation capability of SAM to medical image segmentation. In this paper, instead of fine-tuning the SAM model, we propose Med SAM Adapter, which integrates the medical specific domain knowledge to the segmentation model, by a simple yet effective adaptation technique. Although this work is still one of a few to transfer the popular NLP technique Adapter to computer vision cases, this simple implementation shows surprisingly good performance on medical image segmentation. A medical image adapted SAM, which we have dubbed Medical SAM Adapter (MSA), shows superior performance on 19 medical image segmentation tasks with various image modalities including CT, MRI, ultrasound image, fundus image, and dermoscopic images. MSA outperforms a wide range of state-of-the-art (SOTA) medical image segmentation methods, such as nnUNet, TransUNet, UNetr, MedSegDiff, and also outperforms the fully fine-turned MedSAM with a considerable performance gap. Code will be released at: https://github.com/WuJunde/Medical-SAM-Adapter."
---

# Medical SAM Adapter: Adapting Segment Anything Model for Medical Image Segmentation

**Source**: [http://arxiv.org/pdf/2304.12620v6](http://arxiv.org/pdf/2304.12620v6)

**Code**: [https://github.com/WuJunde/Medical-SAM-Adapter](https://github.com/WuJunde/Medical-SAM-Adapter)

**Year**: 2023

**Authors**: Wu, Junde,  Fu, Rao,  Fang, Huihui,  Liu, Yuanpei,  Wang, Zhaowei,  Xu, Yanwu,  Jin, Yueming,  Arbel, Tal

## Description

The Segment Anything Model (SAM) has recently gained popularity in the field of image segmentation. Thanks to its impressive capabilities in all-round segmentation tasks and its prompt-based interface, SAM has sparked intensive discussion within the community. It is even said by many prestigious experts that image segmentation task has been "finished" by SAM. However, medical image segmentation, although an important branch of the image segmentation family, seems not to be included in the scope of Segmenting "Anything". Many individual experiments and recent studies have shown that SAM performs subpar in medical image segmentation. A natural question is how to find the missing piece of the puzzle to extend the strong segmentation capability of SAM to medical image segmentation. In this paper, instead of fine-tuning the SAM model, we propose Med SAM Adapter, which integrates the medical specific domain knowledge to the segmentation model, by a simple yet effective adaptation technique. Although this work is still one of a few to transfer the popular NLP technique Adapter to computer vision cases, this simple implementation shows surprisingly good performance on medical image segmentation. A medical image adapted SAM, which we have dubbed Medical SAM Adapter (MSA), shows superior performance on 19 medical image segmentation tasks with various image modalities including CT, MRI, ultrasound image, fundus image, and dermoscopic images. MSA outperforms a wide range of state-of-the-art (SOTA) medical image segmentation methods, such as nnUNet, TransUNet, UNetr, MedSegDiff, and also outperforms the fully fine-turned MedSAM with a considerable performance gap. Code will be released at: https://github.com/WuJunde/Medical-SAM-Adapter.
