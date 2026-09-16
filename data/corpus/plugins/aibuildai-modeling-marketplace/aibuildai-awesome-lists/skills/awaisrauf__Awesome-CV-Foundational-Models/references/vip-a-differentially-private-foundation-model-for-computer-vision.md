---
title: "ViP: A Differentially Private Foundation Model for Computer Vision"
entry_type: paper
source: "http://arxiv.org/pdf/2306.08842v2"
upstream_list: "awaisrauf/Awesome-CV-Foundational-Models"
year: 2023
authors: "Yaodong Yu,  Maziar Sanjabi,  Yi Ma,  Kamalika Chaudhuri,  Chuan Guo"
code_url: "https://github.com/facebookresearch/ViP-MAE"
description: "Artificial intelligence (AI) has seen a tremendous surge in capabilities thanks to the use of foundation models trained on internet-scale data. On the flip side, the uncurated nature of internet-scale data also poses significant privacy and legal risks, as they often contain personal information or copyrighted material that should not be trained on without permission. In this work, we propose as a mitigation measure a recipe to train foundation vision models with differential privacy (DP) guarantee. We identify masked autoencoders as a suitable learning algorithm that aligns well with DP-SGD, and train ViP -- a Vision transformer with differential Privacy -- under a strict privacy budget of $\\epsilon=8$ on the LAION400M dataset. We evaluate the quality of representation learned by ViP using standard downstream vision tasks; in particular, ViP achieves a (non-private) linear probing accuracy of $55.7\\%$ on ImageNet, comparable to that of end-to-end trained AlexNet (trained and evaluated on ImageNet). Our result suggests that scaling to internet-scale data can be practical for private learning. Code is available at \\url{https://github.com/facebookresearch/ViP-MAE}."
---

# ViP: A Differentially Private Foundation Model for Computer Vision

**Source**: [http://arxiv.org/pdf/2306.08842v2](http://arxiv.org/pdf/2306.08842v2)

**Code**: [https://github.com/facebookresearch/ViP-MAE](https://github.com/facebookresearch/ViP-MAE)

**Year**: 2023

**Authors**: Yaodong Yu,  Maziar Sanjabi,  Yi Ma,  Kamalika Chaudhuri,  Chuan Guo

## Description

Artificial intelligence (AI) has seen a tremendous surge in capabilities thanks to the use of foundation models trained on internet-scale data. On the flip side, the uncurated nature of internet-scale data also poses significant privacy and legal risks, as they often contain personal information or copyrighted material that should not be trained on without permission. In this work, we propose as a mitigation measure a recipe to train foundation vision models with differential privacy (DP) guarantee. We identify masked autoencoders as a suitable learning algorithm that aligns well with DP-SGD, and train ViP -- a Vision transformer with differential Privacy -- under a strict privacy budget of $\epsilon=8$ on the LAION400M dataset. We evaluate the quality of representation learned by ViP using standard downstream vision tasks; in particular, ViP achieves a (non-private) linear probing accuracy of $55.7\%$ on ImageNet, comparable to that of end-to-end trained AlexNet (trained and evaluated on ImageNet). Our result suggests that scaling to internet-scale data can be practical for private learning. Code is available at \url{https://github.com/facebookresearch/ViP-MAE}.
