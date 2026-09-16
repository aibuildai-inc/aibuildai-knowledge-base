---
title: "Uni-Mol"
source: https://github.com/dptech-corp/Uni-Mol
language: Python
license: MIT
install: "pip install unimol-tools"
entry_type: repo
tags: ["3D-conformer", "molecular-pretraining", "ADMET", "multitask", "regression"]
---

# Uni-Mol

**Source**: [dptech-corp/Uni-Mol](https://github.com/dptech-corp/Uni-Mol)

## Description

Uni-Mol is a 3D molecular pretraining framework that learns molecular
representations directly from atom coordinates and types. It was developed
by DP Technology and released in two main versions: Uni-Mol (SE(3)-invariant
transformer) and Uni-Mol2 (scaled up with 84M and 164M parameter variants).
It is commonly used for downstream property prediction, ADMET regression,
and virtual screening.

## Key Features

- 3D conformer-based molecular representation (takes atom positions as input)
- Pretrained on ZINC20 (884M molecules) and PubChem
- Supports regression, classification, and multitask heads via `unimol-tools`
- Built-in scaffold-balanced splitting and k-fold cross-validation
- Supports 84M and 164M model sizes for Uni-Mol2

## Installation

```bash
pip install unimol-tools
```

## Usage Pattern

```python
from unimol_tools import MolTrain, MolPredict

clf = MolTrain(
    task="regression",
    model_name="unimolv2",
    model_size="84m",
    target_cols=["endpoint"],
    smiles_col="SMILES",
    split="scaffold",
)
clf.fit(data="train.csv")
```
