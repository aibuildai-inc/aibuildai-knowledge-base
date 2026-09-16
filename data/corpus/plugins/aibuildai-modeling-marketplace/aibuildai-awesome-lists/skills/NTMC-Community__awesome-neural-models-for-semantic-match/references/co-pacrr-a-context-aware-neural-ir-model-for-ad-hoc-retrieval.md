---
title: "Co-PACRR: A Context-Aware Neural IR Model for Ad-hoc Retrieval."
entry_type: paper
source: "https://arxiv.org/pdf/1706.10192.pdf"
upstream_list: "NTMC-Community/awesome-neural-models-for-semantic-match"
category: "Ad-hoc Information Retrieval"
venue: "WSDM 2018"
year: "2018"
description: "Co-PACRR takes tow matrices as input, namely $sim_{|q|\\times |d|}$ and $querysim_{|d|}$. Then, several 2D convolutional kernels are first applied to the similarity matrices, and max pooling is applied to the filters. Following this, $n_s$-max pooling captures the strongest $n_s$ signals, and the context similarity corresponding to each term is also appended. Finally, the query term's normalized IDFs are appended, and a feed forward network is applied to obtain the final relevance score."
---

# Co-PACRR: A Context-Aware Neural IR Model for Ad-hoc Retrieval.

**Source**: [https://arxiv.org/pdf/1706.10192.pdf](https://arxiv.org/pdf/1706.10192.pdf)

**Year**: 2018 | **Venue**: WSDM 2018 | **Category**: Ad-hoc Information Retrieval

## Description

Co-PACRR takes tow matrices as input, namely $sim_{|q|\times |d|}$ and $querysim_{|d|}$. Then, several 2D convolutional kernels are first applied to the similarity matrices, and max pooling is applied to the filters. Following this, $n_s$-max pooling captures the strongest $n_s$ signals, and the context similarity corresponding to each term is also appended. Finally, the query term's normalized IDFs are appended, and a feed forward network is applied to obtain the final relevance score.
