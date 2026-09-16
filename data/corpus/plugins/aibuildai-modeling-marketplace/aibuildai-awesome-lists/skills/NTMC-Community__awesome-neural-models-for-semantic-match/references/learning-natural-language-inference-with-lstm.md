---
title: "Learning Natural Language Inference with LSTM."
entry_type: paper
source: "http://www.aclweb.org/anthology/N16-1170"
upstream_list: "NTMC-Community/awesome-neural-models-for-semantic-match"
category: "Natural Language Inference"
venue: "NAACL 2016"
year: "2016"
code_url: "https://github.com/NTMC-Community/MatchZoo/blob/master/matchzoo/contrib/models/match_lstm.py"
description: "Match-LSTM uses an LSTM to perform word-by-word matching of the hypothesis with the premise. The LSTM sequentially processes the hypothesis, and at each position, it tries to match the current word in the hypothesis with an attention-weighted representation of the premise."
---

# Learning Natural Language Inference with LSTM.

**Source**: [http://www.aclweb.org/anthology/N16-1170](http://www.aclweb.org/anthology/N16-1170)

**Code**: [https://github.com/NTMC-Community/MatchZoo/blob/master/matchzoo/contrib/models/match_lstm.py](https://github.com/NTMC-Community/MatchZoo/blob/master/matchzoo/contrib/models/match_lstm.py)

**Year**: 2016 | **Venue**: NAACL 2016 | **Category**: Natural Language Inference

## Description

Match-LSTM uses an LSTM to perform word-by-word matching of the hypothesis with the premise. The LSTM sequentially processes the hypothesis, and at each position, it tries to match the current word in the hypothesis with an attention-weighted representation of the premise.
