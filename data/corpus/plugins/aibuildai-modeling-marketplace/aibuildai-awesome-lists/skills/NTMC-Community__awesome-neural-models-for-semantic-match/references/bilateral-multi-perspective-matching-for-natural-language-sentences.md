---
title: "Bilateral Multi-Perspective Matching for Natural Language Sentences."
entry_type: paper
source: "https://arxiv.org/pdf/1702.03814.pdf"
upstream_list: "NTMC-Community/awesome-neural-models-for-semantic-match"
category: "Natural Language Inference"
venue: "IJCAI 2017"
year: "2017"
code_url: "https://github.com/NTMC-Community/MatchZoo/blob/master/matchzoo/contrib/models/bimpm.py"
description: "Given two sentences P and Q, BiMPM first encodes them with a BiLSTM encoder. Next, it matches the two encoded sentences in two directions P against Q and Q against P. In each matching direction, each time step of one sentence is matched against all timesteps of the other sentence from multiple perspectives. Then, another BiLSTM layer is utilized to aggregate the matching results into a fixed-length matching vector. Finally, based on the matching vector, a decision is made through a fully connected layer."
---

# Bilateral Multi-Perspective Matching for Natural Language Sentences.

**Source**: [https://arxiv.org/pdf/1702.03814.pdf](https://arxiv.org/pdf/1702.03814.pdf)

**Code**: [https://github.com/NTMC-Community/MatchZoo/blob/master/matchzoo/contrib/models/bimpm.py](https://github.com/NTMC-Community/MatchZoo/blob/master/matchzoo/contrib/models/bimpm.py)

**Year**: 2017 | **Venue**: IJCAI 2017 | **Category**: Natural Language Inference

## Description

Given two sentences P and Q, BiMPM first encodes them with a BiLSTM encoder. Next, it matches the two encoded sentences in two directions P against Q and Q against P. In each matching direction, each time step of one sentence is matched against all timesteps of the other sentence from multiple perspectives. Then, another BiLSTM layer is utilized to aggregate the matching results into a fixed-length matching vector. Finally, based on the matching vector, a decision is made through a fully connected layer.
