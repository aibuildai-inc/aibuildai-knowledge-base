---
title: "AugLiChem"
source: https://github.com/BaratiLab/AugLiChem
language: Python
license: MIT
install: "pip install auglichem"
entry_type: repo
tags: ["augmentation", "SMILES", "molecular-graph", "contrastive-learning"]
---

# AugLiChem

**Source**: [BaratiLab/AugLiChem](https://github.com/BaratiLab/AugLiChem)

## Description

AugLiChem is a molecular data augmentation library that provides
SMILES-level and graph-level transformations for chemical machine
learning. It was developed to support contrastive learning and data
augmentation workflows for molecular property prediction, improving
generalization on small chemical datasets.

## Key Features

- SMILES-level augmentations (randomization, substructure masking)
- Graph-level augmentations (node dropping, edge perturbation, subgraph sampling)
- Designed for contrastive pretraining frameworks
- Compatible with PyTorch Geometric and PyTorch

## Installation

```bash
pip install auglichem
```

## Usage Pattern

```python
from auglichem.molecule import Compose, RandomAtomMask, RandomBondDelete

transform = Compose([RandomAtomMask(p=0.1), RandomBondDelete(p=0.1)])
augmented_graph = transform(molecular_graph)
```
