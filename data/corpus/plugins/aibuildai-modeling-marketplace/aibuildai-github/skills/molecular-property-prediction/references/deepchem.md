---
title: "DeepChem"
source: https://github.com/deepchem/deepchem
language: Python
license: MIT
install: "pip install deepchem"
entry_type: repo
tags: ["cheminformatics", "general-purpose", "ML-framework", "featurization", "datasets"]
---

# DeepChem

**Source**: [deepchem/deepchem](https://github.com/deepchem/deepchem)

## Description

DeepChem is a long-standing open-source framework for applying machine
learning to chemistry, biology, and materials science. It provides
molecular featurizers, model implementations (graph neural networks,
transformers, classical ML), dataset loaders, and utilities for training
and evaluation. It covers a broader scope than property prediction alone,
including docking, generative models, and quantum chemistry.

## Key Features

- Wide library of molecular featurizers (ECFP, GraphConv, MolGraphConvFeaturizer, etc.)
- Model zoo including GNNs, transformers, random forests, and SVMs
- Built-in loaders for MoleculeNet benchmark datasets
- Supports regression, classification, and generative tasks
- Integration with RDKit, PyTorch, and TensorFlow backends

## Installation

```bash
pip install deepchem
```

## Usage Pattern

```python
import deepchem as dc

tasks, datasets, transformers = dc.molnet.load_delaney()
train, valid, test = datasets
model = dc.models.GraphConvModel(n_tasks=len(tasks), mode="regression")
model.fit(train)
```
