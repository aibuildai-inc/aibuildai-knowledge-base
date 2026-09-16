---
title: "admet-ai"
source: https://github.com/swansonk14/admet_ai
language: Python
license: MIT
install: "pip install admet-ai"
entry_type: repo
tags: ["ADMET-prediction", "feature-extraction", "pretrained-models", "chemprop"]
---

# admet-ai

**Source**: [swansonk14/admet_ai](https://github.com/swansonk14/admet_ai)

## Description

ADMET-AI is a collection of pretrained molecular property prediction
models covering 41 ADMET endpoints, developed at Stanford's MAPLE lab.
It wraps Chemprop models trained on Therapeutics Data Commons ADMET
benchmarks and provides a simple Python API and web interface for making
predictions on new molecules. Commonly used as a feature generator for
downstream tasks.

## Key Features

- 41 pretrained ADMET prediction endpoints
- One-call prediction from SMILES (`AdmetPredictor().predict(smiles)`)
- Uses Chemprop D-MPNN under the hood
- Returns numerical predictions usable as auxiliary features
- Includes web interface (`admet_web`) for interactive use

## Installation

```bash
pip install admet-ai
```

## Usage Pattern

```python
from admet_ai import ADMETModel

model = ADMETModel()
preds = model.predict(smiles=["CCO", "c1ccccc1"])
# preds is a DataFrame with 41 columns
```
