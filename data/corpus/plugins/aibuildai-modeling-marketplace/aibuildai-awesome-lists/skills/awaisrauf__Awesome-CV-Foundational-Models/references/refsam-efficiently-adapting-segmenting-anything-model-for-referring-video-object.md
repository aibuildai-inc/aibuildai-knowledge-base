---
title: "RefSAM: Efficiently Adapting Segmenting Anything Model for Referring Video Object Segmentation"
entry_type: paper
source: "http://arxiv.org/pdf/2307.00997v1"
upstream_list: "awaisrauf/Awesome-CV-Foundational-Models"
year: 2023
authors: "Yonglin, Li,  Jing, Zhang,  Xiao, Teng,  Long, Lan"
code_url: "https://github.com/LancasterLi/RefSAM"
description: "The Segment Anything Model (SAM) has gained significant attention for its impressive performance in image segmentation. However, it lacks proficiency in referring video object segmentation (RVOS) due to the need for precise user-interactive prompts and limited understanding of different modalities, such as language and vision. This paper presents the RefSAM model, which for the first time explores the potential of SAM for RVOS by incorporating multi-view information from diverse modalities and successive frames at different timestamps. Our proposed approach adapts the original SAM model to enhance cross-modality learning by employing a lightweight Cross-Modal MLP that projects the text embedding of the referring expression into sparse and dense embeddings, serving as user-interactive prompts. Subsequently, a parameter-efficient tuning strategy is employed to effectively align and fuse the language and vision features. Through comprehensive ablation studies, we demonstrate the practical and effective design choices of our strategy. Extensive experiments conducted on Ref-Youtu-VOS and Ref-DAVIS17 datasets validate the superiority and effectiveness of our RefSAM model over existing methods. The code and models will be made publicly at \\href{https://github.com/LancasterLi/RefSAM}{github.com/LancasterLi/RefSAM}."
---

# RefSAM: Efficiently Adapting Segmenting Anything Model for Referring Video Object Segmentation

**Source**: [http://arxiv.org/pdf/2307.00997v1](http://arxiv.org/pdf/2307.00997v1)

**Code**: [https://github.com/LancasterLi/RefSAM](https://github.com/LancasterLi/RefSAM)

**Year**: 2023

**Authors**: Yonglin, Li,  Jing, Zhang,  Xiao, Teng,  Long, Lan

## Description

The Segment Anything Model (SAM) has gained significant attention for its impressive performance in image segmentation. However, it lacks proficiency in referring video object segmentation (RVOS) due to the need for precise user-interactive prompts and limited understanding of different modalities, such as language and vision. This paper presents the RefSAM model, which for the first time explores the potential of SAM for RVOS by incorporating multi-view information from diverse modalities and successive frames at different timestamps. Our proposed approach adapts the original SAM model to enhance cross-modality learning by employing a lightweight Cross-Modal MLP that projects the text embedding of the referring expression into sparse and dense embeddings, serving as user-interactive prompts. Subsequently, a parameter-efficient tuning strategy is employed to effectively align and fuse the language and vision features. Through comprehensive ablation studies, we demonstrate the practical and effective design choices of our strategy. Extensive experiments conducted on Ref-Youtu-VOS and Ref-DAVIS17 datasets validate the superiority and effectiveness of our RefSAM model over existing methods. The code and models will be made publicly at \href{https://github.com/LancasterLi/RefSAM}{github.com/LancasterLi/RefSAM}.
