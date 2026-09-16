---
description: >-
  ML playbook for molecular property materials competitions. Use when tackling a Kaggle-style competition involving molecular property materials. Teaches how to reason about route the problem by input–output geometry before choosing a model, preserve chemical information at the representation boundary, model whole-molecule context at the target’s native granularity, treat invariance as a testable inductive bias, not a dogma. 12 solution writeups from NeurIPS Open Polymer Prediction 2025.
---

# Molecular Property & Materials Competition Playbook

Molecular-materials competitions can look similar scientifically while exposing different learning interfaces: SMILES strings with sparse multi-property labels, molecular graphs, 3D conformers that can fail to generate, and descriptor tables. The central challenge is to preserve the chemically relevant information and symmetries while building validation, training, and inference procedures that reflect the metric and the deployment distribution.

**Source material:** 12 solution writeups from NeurIPS Open Polymer Prediction 2025.

## Principle Index

| # | Principle | Consensus | File |
|---|-----------|-----------|------|
| 1 | Route the Problem by Input–Output Geometry Before Choosing a Model | The 12 polymer writeups spanned SMILES language models, graph neural networks, 3D conformer models, and descriptor-based tabular models; the winning solution combined three of these families rather than committing to one. | principles/01.md |
| 2 | Preserve Chemical Information at the Representation Boundary | 8/12 polymer writeups used fingerprints or descriptors that preserved chemical information at the representation boundary. | principles/02.md |
| 3 | Model Whole-Molecule Context at the Target’s Native Granularity | Several polymer writeups encoded context beyond local substructures: a global graph node, repeat-unit chain extension, attachment-point position features, or 3D shape descriptors. | principles/03.md |
| 4 | Treat Invariance as a Testable Inductive Bias, Not a Dogma | Polymer solutions handled notation invariance in different ways: SMILES language models used randomized SMILES in training and test-time averaging, graph models were invariant by construction, and some notation variants gave no gain. | principles/04.md |
| 5 | Combine Learned Representations with Complementary Chemical Features | 8/12 polymer writeups explicitly used fingerprints or descriptors, while strong minimal-input solutions demonstrated that such features must earn their place. | principles/05.md |
| 6 | Allocate Capacity According to Target and Group Heterogeneity | 7/12 polymer writeups explicitly used property-specific models or processing. | principles/06.md |
| 7 | Validate by the Deployment Unit, Chemical Neighborhood, and Error Slice | 8/12 polymer writeups explicitly discussed folds, validation, grouping, clustering, similarity, or deduplication. | principles/07.md |
| 8 | Reconcile External Data Before Scaling Its Volume | 8/12 polymer writeups used or discussed external datasets. | principles/08.md |
| 9 | Use Augmentation to Match Real Nuisance Variation Without Changing Chemistry | Polymer writeups used randomized SMILES enumeration, repeat-unit chain extension, or targeted substituent changes, while other notation variants and a mixture-model augmentation from public notebooks gave no gain. | principles/09.md |
| 10 | Add Auxiliary Supervision Only When It Encodes the Same Mechanism | Polymer writeups reported gains from ranking pretraining on pseudolabels and from simulation-derived features stacked into a tabular model, while contrastive pretraining did not help; reported gains were highly conditional. | principles/10.md |
| 11 | Optimize the Metric’s Statistical Functional, Not a Convenient Proxy | Polymer solutions trained or selected models on the weighted multi-property MAE, and several replaced MSE with a metric-matched, Huber, or quantile objective. | principles/11.md |
| 12 | Use Curriculum and Scaling Only After the Data Path Is Stable | A recurring pattern across the top solutions: polymer winners used staged pretraining and fine-tuning schedules. | principles/12.md |
| 13 | Ensemble Orthogonal Error Sources, Not Merely More Seeds | 9/12 polymer writeups mentioned ensembling, blending, or stacking. | principles/13.md |
| 14 | Use Chemical Validity as a Routing Constraint, Not as Proof of Correctness | Polymer writeups deduplicated data by canonical SMILES, and one 3D-model solution routed molecules that failed conformer generation to a descriptor model, which its authors called key to their result. | principles/14.md |
| 15 | Calibrate Predictions Only After Diagnosing a Reproducible Shift | 9/12 polymer writeups discussed post-processing, calibration, bias, shifts, or rescaling. | principles/15.md |

## How to Use

Use Glob to list principles/*.md, then Read the principles relevant to your task. The references/ directory holds the raw ranked writeups these principles were summarized from.

Apply these principles as a decision tree: route by representation and target granularity, audit information preservation and symmetry, establish leakage-resistant validation, then add data, supervision, scale, and ensembles only when their mechanisms survive that validation. Reserve calibration and validity-based routing for diagnosed failure modes rather than treating them as universal tricks.
