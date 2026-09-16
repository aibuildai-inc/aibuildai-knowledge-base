---
title: "TDC (Therapeutics Data Commons)"
source: https://github.com/mims-harvard/TDC
language: Python
license: MIT
install: "pip install PyTDC"
entry_type: repo
tags: ["therapeutic-data", "benchmark", "ADMET", "drug-discovery", "datasets"]
---

# TDC (Therapeutics Data Commons)

**Source**: [mims-harvard/TDC](https://github.com/mims-harvard/TDC)

## Description

Therapeutics Data Commons (TDC) is an open-science initiative out of
Harvard MIMS that provides a unified Python interface to machine learning
datasets and benchmarks across therapeutic modalities. It covers
single-instance prediction (ADME, toxicity, HTS), multi-instance prediction
(drug-target interaction, drug-drug interaction), and generation tasks.
The PyTDC SDK is the primary access mechanism.

## Key Features

- Unified SDK for 70+ therapeutic ML datasets
- One-line dataset loading (`ADME(name='Caco2_Wang')`)
- Built-in scaffold-balanced and random train/valid/test splits
- ADMET benchmark group with leaderboards
- Covers ADME, Tox, HTS, protein-ligand, drug-response tasks

## Installation

```bash
pip install PyTDC
```

## Usage Pattern

```python
from tdc.single_pred import ADME

data = ADME(name="Caco2_Wang")
split = data.get_split(method="scaffold")
train_df = split["train"]
valid_df = split["valid"]
test_df = split["test"]
```
