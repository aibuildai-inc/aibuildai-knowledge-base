---
title: "aNMM: Ranking Short Answer Texts with Attention-Based Neural Matching Model."
entry_type: paper
source: "https://arxiv.org/pdf/1801.01641.pdf"
upstream_list: "NTMC-Community/awesome-neural-models-for-semantic-match"
category: "Community Question Answer"
venue: "CIKM 2016"
year: "2016"
code_url: "https://github.com/NTMC-Community/MatchZoo/blob/master/matchzoo/models/anmm.py"
description: "The aNMM referes to attention-based Neural Matching Model. In an abstract level, it contains three steps: 1) it construct QA matching matrix for each question and answer pair with pre-trained word embeddings. 2) it then employ a deep neural network with value-shared weighting scheme in the first lyaer, and fully connected layer in the rest to learn hierarchical abstraction of the semantic matching between question and answer terms. 3) finally, it employs a question attention network to learn question term importance and produce the final ranking score."
---

# aNMM: Ranking Short Answer Texts with Attention-Based Neural Matching Model.

**Source**: [https://arxiv.org/pdf/1801.01641.pdf](https://arxiv.org/pdf/1801.01641.pdf)

**Code**: [https://github.com/NTMC-Community/MatchZoo/blob/master/matchzoo/models/anmm.py](https://github.com/NTMC-Community/MatchZoo/blob/master/matchzoo/models/anmm.py)

**Year**: 2016 | **Venue**: CIKM 2016 | **Category**: Community Question Answer

## Description

The aNMM referes to attention-based Neural Matching Model. In an abstract level, it contains three steps: 1) it construct QA matching matrix for each question and answer pair with pre-trained word embeddings. 2) it then employ a deep neural network with value-shared weighting scheme in the first lyaer, and fully connected layer in the rest to learn hierarchical abstraction of the semantic matching between question and answer terms. 3) finally, it employs a question attention network to learn question term importance and produce the final ranking score.
