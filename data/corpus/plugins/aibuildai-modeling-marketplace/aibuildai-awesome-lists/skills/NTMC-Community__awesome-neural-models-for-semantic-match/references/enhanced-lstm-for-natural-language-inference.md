---
title: "Enhanced LSTM for Natural Language Inference."
entry_type: paper
source: "https://arxiv.org/pdf/1609.06038.pdf"
upstream_list: "NTMC-Community/awesome-neural-models-for-semantic-match"
category: "Natural Language Inference"
venue: "ACL 2017"
year: "2017"
description: "ESIM explicitly encodes parsing information with recursive networks in both local inference modeling and inference composition. In a high-level view, the model consists of four components, namely input encoding, local inference modeling, inference composition, and prediction. The input encoding takes BiLSTM as well as tree-LSTM on the inputs (i.e., premise and hypothesis). Then, two separate local inference modeling components are built on these two encoding outputs. Next, it performs the composition sequentially or in its parse context using BiLSTM and tree-LSTM, respectively. Finally, the multilayer perceptron is applied to predict the final output."
---

# Enhanced LSTM for Natural Language Inference.

**Source**: [https://arxiv.org/pdf/1609.06038.pdf](https://arxiv.org/pdf/1609.06038.pdf)

**Year**: 2017 | **Venue**: ACL 2017 | **Category**: Natural Language Inference

## Description

ESIM explicitly encodes parsing information with recursive networks in both local inference modeling and inference composition. In a high-level view, the model consists of four components, namely input encoding, local inference modeling, inference composition, and prediction. The input encoding takes BiLSTM as well as tree-LSTM on the inputs (i.e., premise and hypothesis). Then, two separate local inference modeling components are built on these two encoding outputs. Next, it performs the composition sequentially or in its parse context using BiLSTM and tree-LSTM, respectively. Finally, the multilayer perceptron is applied to predict the final output.
