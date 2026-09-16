---
title: "MultiGranCNN: An Architecture for General Matching of Text Chunks on Multiple Levels of Granularity."
entry_type: paper
source: "https://aclanthology.info/pdf/P/P15/P15-1007.pdf"
upstream_list: "NTMC-Community/awesome-neural-models-for-semantic-match"
category: "Paraphrase Identification"
venue: "ACL 2015"
year: "2015"
description: "MultiGranCNN firstly utilize gpCNN to learn representations for generalized phrases where a generalized phrase is a general term for subsequences of all granularities: words, short phrases, long phrases and the sentence itself. Then, one of three match feature models (DIRECTSIM, CONCAT or INDIRECTSIM) produces an $s_1 \\times s_2$ match feature matrix which is further reduced to a fixed size matrix by dynamic 2D pooing. Finally, the mfCNN extracts interaction features of increasing complexity from the basic interaction features computed by the match feature model. Finally, the output of the last block of mfCNN is the input to an MLP that computes the match score."
---

# MultiGranCNN: An Architecture for General Matching of Text Chunks on Multiple Levels of Granularity.

**Source**: [https://aclanthology.info/pdf/P/P15/P15-1007.pdf](https://aclanthology.info/pdf/P/P15/P15-1007.pdf)

**Year**: 2015 | **Venue**: ACL 2015 | **Category**: Paraphrase Identification

## Description

MultiGranCNN firstly utilize gpCNN to learn representations for generalized phrases where a generalized phrase is a general term for subsequences of all granularities: words, short phrases, long phrases and the sentence itself. Then, one of three match feature models (DIRECTSIM, CONCAT or INDIRECTSIM) produces an $s_1 \times s_2$ match feature matrix which is further reduced to a fixed size matrix by dynamic 2D pooing. Finally, the mfCNN extracts interaction features of increasing complexity from the basic interaction features computed by the match feature model. Finally, the output of the last block of mfCNN is the input to an MLP that computes the match score.
