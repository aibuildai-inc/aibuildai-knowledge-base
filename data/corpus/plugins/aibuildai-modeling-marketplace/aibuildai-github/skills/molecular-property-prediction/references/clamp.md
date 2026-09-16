---
title: "CLAMP"
source: https://github.com/ml-jku/clamp
language: Python
license: GPL-3.0
install: "See repo README"
entry_type: repo
tags: ["contrastive-learning", "assay-aware", "foundation-model", "ICML-2023"]
---

# CLAMP

**Source**: [ml-jku/clamp](https://github.com/ml-jku/clamp)

## Description

CLAMP (Contrastive Language-Assay-Molecule Pretraining) is a foundation
model developed at the Institute for Machine Learning, JKU Linz (Sepp
Hochreiter's group). It learns joint molecule-assay representations using
contrastive pretraining over biological assay descriptions, enabling
zero-shot and few-shot bioactivity prediction. Published at ICML 2023.

## Key Features

- Contrastive pretraining over molecule-assay pairs
- Joint embedding space aligning molecular structures with assay descriptions
- Supports zero-shot bioactivity prediction from natural language queries
- Pretrained on public ChEMBL bioactivity data

## Installation

```bash
git clone https://github.com/ml-jku/clamp
cd clamp
pip install -e .
```

See the repository README for dataset setup and pretrained checkpoint
download instructions.
