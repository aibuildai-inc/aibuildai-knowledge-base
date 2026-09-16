---
title: "SAM-DA: UAV Tracks Anything at Night with SAM-Powered Domain Adaptation"
entry_type: paper
source: "http://arxiv.org/pdf/2307.01024v1"
upstream_list: "awaisrauf/Awesome-CV-Foundational-Models"
year: 2023
authors: "Liangliang, Yao,  Haobo, Zuo,  Guangze, Zheng,  Changhong, Fu,  Jia, Pan"
code_url: "https://github.com/vision4robotics/SAM-DA"
description: "Domain adaptation (DA) has demonstrated significant promise for real-time nighttime unmanned aerial vehicle (UAV) tracking. However, the state-of-the-art (SOTA) DA still lacks the potential object with accurate pixel-level location and boundary to generate the high-quality target domain training sample. This key issue constrains the transfer learning of the real-time daytime SOTA trackers for challenging nighttime UAV tracking. Recently, the notable Segment Anything Model (SAM) has achieved remarkable zero-shot generalization ability to discover abundant potential objects due to its huge data-driven training approach. To solve the aforementioned issue, this work proposes a novel SAM-powered DA framework for real-time nighttime UAV tracking, i.e., SAM-DA. Specifically, an innovative SAM-powered target domain training sample swelling is designed to determine enormous high-quality target domain training samples from every single raw nighttime image. This novel one-to-many method significantly expands the high-quality target domain training sample for DA. Comprehensive experiments on extensive nighttime UAV videos prove the robustness and domain adaptability of SAM-DA for nighttime UAV tracking. Especially, compared to the SOTA DA, SAM-DA can achieve better performance with fewer raw nighttime images, i.e., the fewer-better training. This economized training approach facilitates the quick validation and deployment of algorithms for UAVs. The code is available at https://github.com/vision4robotics/SAM-DA."
---

# SAM-DA: UAV Tracks Anything at Night with SAM-Powered Domain Adaptation

**Source**: [http://arxiv.org/pdf/2307.01024v1](http://arxiv.org/pdf/2307.01024v1)

**Code**: [https://github.com/vision4robotics/SAM-DA](https://github.com/vision4robotics/SAM-DA)

**Year**: 2023

**Authors**: Liangliang, Yao,  Haobo, Zuo,  Guangze, Zheng,  Changhong, Fu,  Jia, Pan

## Description

Domain adaptation (DA) has demonstrated significant promise for real-time nighttime unmanned aerial vehicle (UAV) tracking. However, the state-of-the-art (SOTA) DA still lacks the potential object with accurate pixel-level location and boundary to generate the high-quality target domain training sample. This key issue constrains the transfer learning of the real-time daytime SOTA trackers for challenging nighttime UAV tracking. Recently, the notable Segment Anything Model (SAM) has achieved remarkable zero-shot generalization ability to discover abundant potential objects due to its huge data-driven training approach. To solve the aforementioned issue, this work proposes a novel SAM-powered DA framework for real-time nighttime UAV tracking, i.e., SAM-DA. Specifically, an innovative SAM-powered target domain training sample swelling is designed to determine enormous high-quality target domain training samples from every single raw nighttime image. This novel one-to-many method significantly expands the high-quality target domain training sample for DA. Comprehensive experiments on extensive nighttime UAV videos prove the robustness and domain adaptability of SAM-DA for nighttime UAV tracking. Especially, compared to the SOTA DA, SAM-DA can achieve better performance with fewer raw nighttime images, i.e., the fewer-better training. This economized training approach facilitates the quick validation and deployment of algorithms for UAVs. The code is available at https://github.com/vision4robotics/SAM-DA.
