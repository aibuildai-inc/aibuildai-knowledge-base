---
title: "A Deep Relevance Matching Model for Ad-hoc Retrieval (*A variation of DRMM)."
entry_type: paper
source: "https://link.springer.com/chapter/10.1007/978-3-030-01012-6_2"
upstream_list: "NTMC-Community/awesome-neural-models-for-semantic-match"
category: "Ad-hoc Information Retrieval"
venue: "CCIR 2018"
year: "2018"
code_url: "https://github.com/NTMC-Community/MatchZoo/blob/master/matchzoo/models/drmm_tks.py"
description: "DRMM_TKS is an variant version of DRMM, where the matching histogram layer is replaced by a sorted top-k pooling layer. Specifically, each query term is interacted with all the document terms to produce the query term-level interaction vector. Then, a sorted top-k pooling layer is applied on this vector to obtain the fixed length interaction vector. All the other components remains fixed with original DRMM."
---

# A Deep Relevance Matching Model for Ad-hoc Retrieval (*A variation of DRMM).

**Source**: [https://link.springer.com/chapter/10.1007/978-3-030-01012-6_2](https://link.springer.com/chapter/10.1007/978-3-030-01012-6_2)

**Code**: [https://github.com/NTMC-Community/MatchZoo/blob/master/matchzoo/models/drmm_tks.py](https://github.com/NTMC-Community/MatchZoo/blob/master/matchzoo/models/drmm_tks.py)

**Year**: 2018 | **Venue**: CCIR 2018 | **Category**: Ad-hoc Information Retrieval

## Description

DRMM_TKS is an variant version of DRMM, where the matching histogram layer is replaced by a sorted top-k pooling layer. Specifically, each query term is interacted with all the document terms to produce the query term-level interaction vector. Then, a sorted top-k pooling layer is applied on this vector to obtain the fixed length interaction vector. All the other components remains fixed with original DRMM.
