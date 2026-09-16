---
description: >-
  ML playbook for molecular biochemical prediction competitions. Use when tackling a Kaggle-style competition involving molecular biochemical prediction. Teaches how to reason about choose molecular representation by data availability and property type, design cross-validation to match the generalization challenge, integrate external data with label rescaling and error-based filtering, apply multi-stage training when non-scored auxiliary tasks exist. 89 top-solution writeups across 9 competitions: BELKA (drug binding), MoA (mechanism of action), CAFA-5 (protein function), Novozymes (enzyme stability), polymer properties, single-cell perturbations, RNA folding competitions
---

# Molecular Biochemical Prediction Playbook

Molecular biochemical prediction tasks require predicting physical, chemical, or biological properties of molecules (small molecules, proteins, RNA, polymers) from their structure. The core challenge is choosing molecular representations that capture the right level of structural detail while being learnable from limited labeled data, and handling the severe data quality issues (label noise, distribution shifts, small datasets) endemic to experimental biochemical measurements.

**Source material:** 89 top-solution writeups across 9 competitions: BELKA (drug binding), MoA (mechanism of action), CAFA-5 (protein function), Novozymes (enzyme stability), polymer properties, single-cell perturbations, RNA folding competitions

## Principle Index

| # | Principle | Consensus | File |
|---|-----------|-----------|------|
| 1 | Choose Molecular Representation by Data Availability and Property Type | All top solutions prioritize this decision and match the representation to their data regime | principles/01.md |
| 2 | Design Cross-Validation to Match the Generalization Challenge | 9/9 competitions; every top solution used domain-specific CV, never random splits | principles/02.md |
| 3 | Integrate External Data with Label Rescaling and Error-Based Filtering | Most winning solutions used external data; all that did applied cleaning - raw merging failed | principles/03.md |
| 4 | Apply Multi-Stage Training When Non-Scored Auxiliary Tasks Exist | Several MoA top solutions pretrained on the non-scored targets and transferred to the scored targets; directly predicting scored targets left performance on table | principles/04.md |
| 5 | Use Test-Time Augmentation for Sequence-Based Models via SMILES Randomization | 4/5 top transformer-based solutions used TTA; single-prediction transformers left 0.01 LB improvement unused | principles/05.md |
| 6 | Pretrain Transformers on Pseudolabeled Domain Data When Foundation Models Underperform | 3/5 transformer solutions added domain pretraining; using only HuggingFace weights was suboptimal | principles/06.md |
| 7 | Combine Molecular Fingerprints with Learned Representations for Tabular Models | 6/6 top GBDT/tabular solutions combined fingerprints + descriptors + learned embeddings | principles/07.md |
| 8 | Handle Severe Class Imbalance with Label Smoothing and Reweighting, Not Resampling | 8/8 solutions on imbalanced tasks (MoA, BELKA) used label smoothing + reweighting; none used SMOTE/oversampling | principles/08.md |
| 9 | Detect and Correct Distribution Shifts in Test Labels via Probing | 1/1 competition with a known label issue (polymer Tg); probing caught shifts random CV missed | principles/09.md |
| 10 | Leverage Domain-Specific Structural Features for Protein and Enzyme Tasks | 5/5 protein/enzyme solutions used 3D structural features; sequence-only models left performance on table | principles/10.md |
| 11 | Design Ensembles for Maximum Diversity Across Representation Types | All top solutions used ensembles; most explicitly maximized diversity across model types not just seeds | principles/11.md |
| 12 | Use Frequent Checkpointing and Early Stopping on Non-Shared Test Distributions | Solutions facing train/test distribution shift, most clearly on BELKA non-shared building blocks, relied on very early stopping or frequent checkpointing | principles/12.md |
| 13 | Encode Positional and Structural Constraints via Distance Embeddings | Recurring in top solutions on sequence-based RNA tasks: distance/position embeddings, not just sequence | principles/13.md |
| 14 | Apply Sample Normalization for Gene Expression and High-Throughput Screening Data | 3/3 solutions on gene expression (MoA, single-cell) and 1/1 on DEL screening used sample-wise normalization | principles/14.md |

## How to Use

Use Glob to list principles/*.md, then Read the principles relevant to your task. The references/ directory holds the raw ranked writeups these principles were distilled from.

These principles form a decision tree, not a checklist: start with representation choice (principle 1), then validation strategy (2), then decide whether external data (3) or multi-stage training (4) applies, then tune training details (5-13). The most impactful decisions are representation + CV strategy - get those right first before optimizing training.
