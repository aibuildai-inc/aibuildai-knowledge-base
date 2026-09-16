---
description: >-
  ML playbook for single cell multimodal prediction competitions. Use when tackling a Kaggle-style competition involving single cell multimodal prediction. Teaches how to reason about classify the missingness geometry before choosing a model, make validation reproduce the hidden experimental axis, normalize according to the assay, not by habit, combine global low-rank state with modality-specific direct features. 25 solution writeups across 2 Kaggle competitions: 16 from Open Problems – Multimodal Single-Cell Integration and 9 from Open Problems – Single-Cell Perturbations, including 13 top-eight solutions. The evidence contains two distinct methodological clusters: cell-level paired-modality translation and aggregate cell-type–compound matrix completion.
---

# Single-Cell Multimodal Prediction Playbook

Single-cell multimodal competitions combine extremely wide, noisy molecular outputs with structured missingness across modality, time, donor, cell type, or perturbation. The central challenge is not merely fitting a multi-output regressor: it is choosing a representation, validation design, and inductive bias that match the exact axis along which the test measurements are missing.

**Source material:** 25 solution writeups across 2 Kaggle competitions: 16 from Open Problems – Multimodal Single-Cell Integration and 9 from Open Problems – Single-Cell Perturbations, including 13 top-eight solutions. The evidence contains two distinct methodological clusters: cell-level paired-modality translation and aggregate cell-type–compound matrix completion.

## Principle Index

| # | Principle | Consensus | File |
|---|-----------|-----------|------|
| 1 | Classify the Missingness Geometry Before Choosing a Model | 2/2 competitions required different solution families; all 13 top-eight writeups reflected this distinction in their representations or model design. | principles/01.md |
| 2 | Make Validation Reproduce the Hidden Experimental Axis | 13/13 top-eight writeups described a deliberate validation scheme; at least 8/13 explicitly held out donor, day, cell type, compound, or adversarially test-like rows. | principles/02.md |
| 3 | Normalize According to the Assay, Not by Habit | 7/8 top multimodal solutions explicitly relied on assay-aware normalization or multiple normalized views; perturbation leaders mostly modeled the provided aggregate response instead of reprocessing raw counts. | principles/03.md |
| 4 | Combine Global Low-Rank State with Modality-Specific Direct Features | All 8 top multimodal writeups used or discussed dimensional reduction; all 8 retained selected direct genes for CITEseq or explicitly reported that direct features did not help Multiome. | principles/04.md |
| 5 | Turn Sparse Entity Labels into Interaction Priors | 5/5 top-eight perturbation solutions learned categorical/entity representations; 3/5 explicitly used target means, medians, standard deviations, or quantiles as input features. | principles/05.md |
| 6 | Compress Targets Only When Their Effective Rank Justifies It | At least 6/8 top multimodal solutions compressed the 23,418-dimensional Multiome target; only 1/5 top perturbation solutions used target SVD in its final system, while another explicitly found full-target regression better. | principles/06.md |
| 7 | Use the Simplest Architecture That Captures Output Coupling | 12/13 top-eight solutions used neural multi-output or factorized models, usually alongside simpler models; plain MLPs appeared in nearly every multimodal ensemble and in several perturbation systems. | principles/07.md |
| 8 | Align Both Loss and Target Space with the Evaluation Metric | At least 8/13 top-eight solutions explicitly used correlation loss, MRRMSE, Huber/MAE mixtures, or another metric-aware objective rather than relying only on raw MSE. | principles/08.md |
| 9 | Regularize in Ways Consistent with the Representation | A minority of 4/13 top-eight solutions reported meaningful gains from mixup, feature masking, c-mixup, or pseudolabeling; several others explicitly found pseudo-labels or generic noise ineffective. | principles/09.md |
| 10 | Bias Training Toward the Deployment Domain Without Throwing Away Signal | Only about 3/13 top-eight solutions explicitly used recency-only training, batch fine-tuning, or domain weighting, but the gains were material when temporal or cell-type drift was strong. | principles/10.md |
| 11 | Ensemble OOF-Verified Error Modes, Not Model Names | 12/13 top-eight solutions used averaging, seed bagging, weighted blending, stacking, or pseudolabel ensembles; the main exception achieved seventh with a single factorized model. | principles/11.md |
| 12 | Postprocess Only Along Proven Metric Invariances | At least 4/13 top-eight solutions explicitly used row normalization, zero-target restoration, clipping, or calibrated output scaling; leaderboard-only scaling produced a notable public-to-private drop. | principles/12.md |

## How to Use

Use Glob to list principles/*.md, then Read the principles relevant to your task. The references/ directory holds the raw ranked writeups these principles were summarized from.

Use the principles as a decision chain: identify the missing axis, validate it faithfully, build an assay- or entity-appropriate representation, and only then choose target geometry, model, loss, and regularization. Let hard-fold OOF evidence govern ensembling and calibration; in these competitions, faithful simulation of deployment was often more valuable than architectural novelty.
