---
title: "Segment Anything Meets Point Tracking"
entry_type: paper
source: "http://arxiv.org/pdf/2307.01197v1"
upstream_list: "awaisrauf/Awesome-CV-Foundational-Models"
year: 2023
authors: "Rajič, Frano,  Ke, Lei,  Tai, Yu-Wing,  Tang, Chi-Keung,  Danelljan, Martin,  Yu, Fisher"
code_url: "https://github.com/SysCV/sam-pt"
description: "The Segment Anything Model (SAM) has established itself as a powerful zero-shot image segmentation model, employing interactive prompts such as points to generate masks. This paper presents SAM-PT, a method extending SAM's capability to tracking and segmenting anything in dynamic videos. SAM-PT leverages robust and sparse point selection and propagation techniques for mask generation, demonstrating that a SAM-based segmentation tracker can yield strong zero-shot performance across popular video object segmentation benchmarks, including DAVIS, YouTube-VOS, and MOSE. Compared to traditional object-centric mask propagation strategies, we uniquely use point propagation to exploit local structure information that is agnostic to object semantics. We highlight the merits of point-based tracking through direct evaluation on the zero-shot open-world Unidentified Video Objects (UVO) benchmark. To further enhance our approach, we utilize K-Medoids clustering for point initialization and track both positive and negative points to clearly distinguish the target object. We also employ multiple mask decoding passes for mask refinement and devise a point re-initialization strategy to improve tracking accuracy. Our code integrates different point trackers and video segmentation benchmarks and will be released at https://github.com/SysCV/sam-pt."
---

# Segment Anything Meets Point Tracking

**Source**: [http://arxiv.org/pdf/2307.01197v1](http://arxiv.org/pdf/2307.01197v1)

**Code**: [https://github.com/SysCV/sam-pt](https://github.com/SysCV/sam-pt)

**Year**: 2023

**Authors**: Rajič, Frano,  Ke, Lei,  Tai, Yu-Wing,  Tang, Chi-Keung,  Danelljan, Martin,  Yu, Fisher

## Description

The Segment Anything Model (SAM) has established itself as a powerful zero-shot image segmentation model, employing interactive prompts such as points to generate masks. This paper presents SAM-PT, a method extending SAM's capability to tracking and segmenting anything in dynamic videos. SAM-PT leverages robust and sparse point selection and propagation techniques for mask generation, demonstrating that a SAM-based segmentation tracker can yield strong zero-shot performance across popular video object segmentation benchmarks, including DAVIS, YouTube-VOS, and MOSE. Compared to traditional object-centric mask propagation strategies, we uniquely use point propagation to exploit local structure information that is agnostic to object semantics. We highlight the merits of point-based tracking through direct evaluation on the zero-shot open-world Unidentified Video Objects (UVO) benchmark. To further enhance our approach, we utilize K-Medoids clustering for point initialization and track both positive and negative points to clearly distinguish the target object. We also employ multiple mask decoding passes for mask refinement and devise a point re-initialization strategy to improve tracking accuracy. Our code integrates different point trackers and video segmentation benchmarks and will be released at https://github.com/SysCV/sam-pt.
