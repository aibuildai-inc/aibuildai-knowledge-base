---
title: "KERMT"
source: https://github.com/Merck/KERMT
language: Python
license: Apache-2.0
install: "See repo README"
entry_type: repo
tags: ["foundation-model", "molecular-representation", "pretraining"]
---

# KERMT

**Source**: [Merck/KERMT](https://github.com/Merck/KERMT)

## Description

KERMT (Knowledge Enhanced Representations for Molecular Tasks) is a
molecular foundation model released by Merck in collaboration with NVIDIA.
It provides pretrained molecular representations that can be fine-tuned
for downstream property prediction tasks. The public release includes
model checkpoints and training code, while industrial versions may use
larger internal datasets.

## Key Features

- Pretrained molecular foundation model
- Supports transfer learning to downstream property prediction
- Released with public checkpoint weights
- Designed for integration into multitask fine-tuning pipelines

## Installation

```bash
git clone https://github.com/Merck/KERMT
cd KERMT
pip install -e .
```

See the repository README for the most up-to-date installation
instructions and dependency requirements.
