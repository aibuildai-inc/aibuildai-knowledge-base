---
title: "Multiway Attention Networks for Modeling Sentences Pairs."
entry_type: paper
source: "https://www.ijcai.org/proceedings/2018/0613.pdf"
upstream_list: "NTMC-Community/awesome-neural-models-for-semantic-match"
category: "Paraphrase Identification"
venue: "IJCAI 2018"
year: "2018"
description: "MwAN is built on the matching-aggregation framework. Given two sentences,  a bidirectional RNN to is applied to obtain contextual word representation of words in two sentences based on the word embeddings. Then it takes four attention functions to match two sentences at the word level. Next, it aggregates the matching information from multiway attention functions with two steps with two bi-directional RNN. Finally, it applies the attention-pooling to the matching information for a fix-length vector and feed it into a multilayer perceptron for the final decision."
---

# Multiway Attention Networks for Modeling Sentences Pairs.

**Source**: [https://www.ijcai.org/proceedings/2018/0613.pdf](https://www.ijcai.org/proceedings/2018/0613.pdf)

**Year**: 2018 | **Venue**: IJCAI 2018 | **Category**: Paraphrase Identification

## Description

MwAN is built on the matching-aggregation framework. Given two sentences,  a bidirectional RNN to is applied to obtain contextual word representation of words in two sentences based on the word embeddings. Then it takes four attention functions to match two sentences at the word level. Next, it aggregates the matching information from multiway attention functions with two steps with two bi-directional RNN. Finally, it applies the attention-pooling to the matching information for a fix-length vector and feed it into a multilayer perceptron for the final decision.
