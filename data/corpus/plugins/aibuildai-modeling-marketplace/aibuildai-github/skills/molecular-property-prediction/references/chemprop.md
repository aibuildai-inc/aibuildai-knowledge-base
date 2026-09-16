---
title: "chemprop"
source: https://github.com/chemprop/chemprop
language: Python
license: MIT
install: "pip install chemprop"
entry_type: repo
tags: ["MPNN", "message-passing", "molecular-graph", "CheMeleon-foundation", "ADMET"]
---

# chemprop

**Source**: [chemprop/chemprop](https://github.com/chemprop/chemprop)

## Description

Chemprop is a message-passing neural network (D-MPNN) framework for
molecular property prediction. Originally developed at MIT CSAIL, it is
widely used for ADMET modeling and SAR analysis. Version 2.x adds
integration with foundation models (CheMeleon) and improved training
infrastructure.

## Key Features

- Directed message-passing neural network (D-MPNN) on molecular graphs
- Foundation model integration via `--from-foundation CHEMELEON` CLI flag
- Built-in scaffold-balanced splitting (`--split SCAFFOLD_BALANCED`)
- Multitask regression and classification support
- Uncertainty estimation and interpretation tools
- CLI-first workflow (`chemprop train`, `chemprop predict`)

## Installation

```bash
pip install chemprop
```

## Usage Pattern

```bash
chemprop train \
    --data-path train.csv \
    --smiles-columns SMILES \
    --target-columns endpoint1 endpoint2 \
    --task-type regression \
    --from-foundation CHEMELEON \
    --split SCAFFOLD_BALANCED \
    --split-sizes 0.9 0.1 0.0
```
