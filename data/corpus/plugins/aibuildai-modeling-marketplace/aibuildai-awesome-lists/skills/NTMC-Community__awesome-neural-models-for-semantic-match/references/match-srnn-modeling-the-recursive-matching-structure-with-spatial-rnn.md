---
title: "Match-SRNN: Modeling the Recursive Matching Structure with Spatial RNN."
entry_type: paper
source: "https://arxiv.org/pdf/1604.04378.pdf"
upstream_list: "NTMC-Community/awesome-neural-models-for-semantic-match"
category: "Paraphrase Identification"
venue: "IJCAI 2016"
year: "2016"
code_url: "https://github.com/NTMC-Community/MatchZoo/blob/master/matchzoo/models/matchsrnn.py"
description: "Match-SRNN view the generation of the global interaction between two texts as a recursive process: i.e. the interaction of two texts at each position is a composition of the interactions between their prefixes as well as the word level interaction at the recurrent position. Firstly, a tensor is constructed to capture the word level interactions. Then a spatial RNN is applied to integrate the local interactions recursively, with importance determined by four types of gates. Finally, the matching score is calculated based on the global interaction."
---

# Match-SRNN: Modeling the Recursive Matching Structure with Spatial RNN.

**Source**: [https://arxiv.org/pdf/1604.04378.pdf](https://arxiv.org/pdf/1604.04378.pdf)

**Code**: [https://github.com/NTMC-Community/MatchZoo/blob/master/matchzoo/models/matchsrnn.py](https://github.com/NTMC-Community/MatchZoo/blob/master/matchzoo/models/matchsrnn.py)

**Year**: 2016 | **Venue**: IJCAI 2016 | **Category**: Paraphrase Identification

## Description

Match-SRNN view the generation of the global interaction between two texts as a recursive process: i.e. the interaction of two texts at each position is a composition of the interactions between their prefixes as well as the word level interaction at the recurrent position. Firstly, a tensor is constructed to capture the word level interactions. Then a spatial RNN is applied to integrate the local interactions recursively, with importance determined by four types of gates. Finally, the matching score is calculated based on the global interaction.
