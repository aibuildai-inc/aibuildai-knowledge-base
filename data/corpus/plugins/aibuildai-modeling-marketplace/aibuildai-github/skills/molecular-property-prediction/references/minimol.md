---
title: "MiniMol"
source: https://github.com/graphcore-research/minimol
language: Python
license: MIT
install: "pip install minimol"
entry_type: repo
tags: ["foundation-model", "lightweight", "molecular-representation"]
---

# MiniMol

**Source**: [graphcore-research/minimol](https://github.com/graphcore-research/minimol)

## Description

MiniMol is a lightweight molecular foundation model released by Graphcore
Research. It is designed to provide competitive molecular representations
at a fraction of the parameter count of larger foundation models, making
it suitable for downstream fine-tuning on limited compute budgets.

## Key Features

- Lightweight foundation model (orders of magnitude smaller than typical LLMs)
- Pretrained on public molecular databases
- Produces fixed-size embeddings usable as features for downstream tasks
- Focused on efficient inference and transfer learning

## Installation

```bash
pip install minimol
```

## Usage Pattern

```python
from minimol import Minimol

model = Minimol()
embeddings = model(["CCO", "c1ccccc1"])  # list of SMILES
```
