---
title: "Learning Deep Structured Semantic Models for Web Search using Clickthrough Data."
entry_type: paper
source: "https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/cikm2013_DSSM_fullversion.pdf"
upstream_list: "NTMC-Community/awesome-neural-models-for-semantic-match"
category: "Ad-hoc Information Retrieval"
venue: "CIKM 2013"
year: "2013"
code_url: "https://github.com/NTMC-Community/MatchZoo/blob/master/matchzoo/models/dssm.py"
description: "DSSM uses a DNN to map high-dimensional sparse text features into low-dimensional dense features in a semantic space. The first hidden layer, with 30k units, accomplish word hashing. The word-hashed features are then projected through multiple layers of non-linear projections. The final layer's neural activities in this DNN from the feature in the semantic spae."
---

# Learning Deep Structured Semantic Models for Web Search using Clickthrough Data.

**Source**: [https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/cikm2013_DSSM_fullversion.pdf](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/cikm2013_DSSM_fullversion.pdf)

**Code**: [https://github.com/NTMC-Community/MatchZoo/blob/master/matchzoo/models/dssm.py](https://github.com/NTMC-Community/MatchZoo/blob/master/matchzoo/models/dssm.py)

**Year**: 2013 | **Venue**: CIKM 2013 | **Category**: Ad-hoc Information Retrieval

## Description

DSSM uses a DNN to map high-dimensional sparse text features into low-dimensional dense features in a semantic space. The first hidden layer, with 30k units, accomplish word hashing. The word-hashed features are then projected through multiple layers of non-linear projections. The final layer's neural activities in this DNN from the feature in the semantic spae.
