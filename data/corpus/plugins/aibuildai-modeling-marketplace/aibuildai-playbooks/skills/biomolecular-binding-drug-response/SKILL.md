---
description: >-
  ML playbook for biomolecular binding drug response competitions. Use when tackling a Kaggle-style competition involving biomolecular binding drug response. Teaches how to reason about diagnose the observation modality, output structure, and metric first, make validation withhold the causal identity that will be novel, match the backbone to the representation instead of chasing generic complexity, share statistical strength across related targets, but test the output parameterization. 21 top-solution writeups across 2 Kaggle competitions: 7 from NeurIPS 2024 BELKA molecular binding prediction (ranks 1–27) and 14 from LISH Mechanisms of Action cellular drug-response prediction (ranks 1–40).
---

# Biomolecular Binding and Drug-Response Prediction Playbook

These competitions share a drug-discovery setting but split into two distinct methodological regimes: structure-to-binding prediction from molecular representations, and perturbation-to-mechanism prediction from high-dimensional cellular assay profiles. The central challenge is to identify what must generalize—new molecular components, unseen compounds, or repeated known compounds—while handling extreme label imbalance and preserving the probability or ranking behavior required by the metric.

**Source material:** 21 top-solution writeups across 2 Kaggle competitions: 7 from NeurIPS 2024 BELKA molecular binding prediction (ranks 1–27) and 14 from LISH Mechanisms of Action cellular drug-response prediction (ranks 1–40).

## Principle Index

| # | Principle | Consensus | File |
|---|-----------|-----------|------|
| 1 | Diagnose the Observation Modality, Output Structure, and Metric First | 7/7 BELKA writeups followed the molecular-structure branch; all detailed MoA solutions followed the cellular-profile branch. Every successful solution was shaped by its competition’s ranking-versus-probability metric. | principles/01.md |
| 2 | Make Validation Withhold the Causal Identity That Will Be Novel | 6/7 BELKA writeups explicitly used or analyzed building-block-aware non-shared validation. 7/14 MoA writeups used, compared, or discussed drug-group-aware validation; several others retained multilabel-stratified CV because it aligned better with their observed leaderboard regime. | principles/02.md |
| 3 | Match the Backbone to the Representation Instead of Chasing Generic Complexity | 7/7 BELKA writeups used molecular sequence, fingerprint, or graph models, with simple SMILES sequence models repeatedly competitive. At least 12/14 MoA writeups used shallow MLP, residual MLP, TabNet, or learned convolutions over assay features. | principles/03.md |
| 4 | Share Statistical Strength Across Related Targets, but Test the Output Parameterization | At least 5/7 BELKA solutions explicitly predicted all three proteins jointly. Most MoA neural solutions used joint multi-label heads, while the 8th-place solution showed that label-powerset multiclass prediction was a useful low-correlation alternative. | principles/04.md |
| 5 | Use Auxiliary Supervision to Teach the Encoder Before Rare Labels | 4/7 BELKA writeups used or investigated molecular pretraining, including the winner’s MLM plus SMILES-to-ECFP stages. At least 7/14 MoA writeups used non-scored targets for pretraining, transfer, stacking, or auxiliary prediction. | principles/05.md |
| 6 | Separate Memorization and Extrapolation When the Test Set Mixes Both | 4/7 BELKA writeups explicitly built different shared/non-shared pipelines. In MoA, the 5th-place solution explicitly gated models for seen versus unseen drugs; several others blended random-split and drug-group-split models to cover both regimes. | principles/06.md |
| 7 | Handle Imbalance Without Changing What Validation Means | 6/7 BELKA writeups explicitly sampled negatives, used focal/weighted losses, or discussed sub-1% positives. At least 10/14 MoA writeups used label smoothing, target weighting, stratification, or another imbalance-aware device. | principles/07.md |
| 8 | Match Regularization and Checkpoint Frequency to the Generalization Regime | 6/7 BELKA writeups reported rapid non-shared overfitting, separate checkpoint policies, or strong sensitivity to depth and training duration. At least 10/14 MoA writeups used early stopping, label smoothing, weight decay, shallow networks, or per-target stopping. | principles/08.md |
| 9 | Treat Fold and Seed Variation as a Diagnostic, Not Just Bagging Material | 5/7 BELKA solutions averaged folds, seeds, epochs, or checkpoints; BELKA writeups repeatedly emphasized severe non-shared variance. At least 11/14 MoA solutions used multiple folds, seeds, or both. | principles/09.md |
| 10 | Ensemble Residual Diversity, Not Model Names | 5/7 BELKA solutions used multi-model or multi-checkpoint ensembles, while the winner and 8th-place solution showed that a strong single Transformer can remain competitive. Essentially all substantive MoA writeups used blending or stacking, often with MLP, TabNet, ResNet, or CNN diversity. | principles/10.md |
| 11 | Post-process According to Metric Geometry and Hard Biological Constraints | BELKA’s ranking metric motivated rank-based or protein-specific blending in several solutions, though only one writeup explicitly used rank ensembling. At least 6/14 MoA writeups explicitly used clipping, smoothing, class-wise blending, control handling, or confidence-aware post-processing; this principle is included because the two metrics demand opposite behavior. | principles/11.md |

## How to Use

Use Glob to list principles/*.md, then Read the principles relevant to your task. The references/ directory holds the raw ranked writeups these principles were summarized from.

Use these principles as a decision sequence: identify modality and novelty, construct validation that reproduces that novelty, choose a matching representation and training objective, and only then regularize, ensemble, and post-process. Keep separate scorecards for interpolation, extrapolation, and target-wise behavior so an aggregate metric never conceals the regime that will determine real-world performance.
