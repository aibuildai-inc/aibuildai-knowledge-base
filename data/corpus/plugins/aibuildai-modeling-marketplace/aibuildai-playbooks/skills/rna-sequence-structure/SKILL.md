---
description: >-
  ML playbook for rna sequence structure competitions. Use when tackling a Kaggle-style competition involving rna sequence structure. Teaches how to reason about choose the modeling regime from the output geometry and metric, represent both backbone neighborhoods and long-range pairing, use a template–neural cascade for 3d prediction, design positional information to extrapolate beyond training lengths. 36 indexed solution writeups across 3 Kaggle competitions: Stanford Ribonanza RNA Folding and Stanford RNA 3D Folding Parts 1 and 2. The synthesis combines detailed review of leading solutions with corpus-wide theme searches, covering chemical reactivity regression, template-based 3D modeling, neural structure prediction, multi-chain complexes, and best-of-five inference.
---

# RNA Sequence–Structure Competition Playbook

RNA sequence–structure tasks split into two fundamentally different regimes: per-nucleotide experimental-profile regression and full 3D coordinate prediction. The common challenge is to combine local backbone context, long-range base-pair interactions, noisy or sparse supervision, and severe distribution shifts in sequence length, homology, release date, and molecular composition.

**Source material:** 36 indexed solution writeups across 3 Kaggle competitions: Stanford Ribonanza RNA Folding and Stanford RNA 3D Folding Parts 1 and 2. The synthesis combines detailed review of leading solutions with corpus-wide theme searches, covering chemical reactivity regression, template-based 3D modeling, neural structure prediction, multi-chain complexes, and best-of-five inference.

## Principle Index

| # | Principle | Consensus | File |
|---|-----------|-----------|------|
| 1 | Choose the Modeling Regime from the Output Geometry and Metric | 3/3 competitions support this split; the profile competition converged on sequence/pair encoders, while both 3D competitions required coordinate-producing or coordinate-transferring pipelines. | principles/01.md |
| 2 | Represent Both Backbone Neighborhoods and Long-Range Pairing | 5/5 top Ribonanza solutions reviewed used an explicit combination of local sequence processing and pair/graph information. | principles/02.md |
| 3 | Use a Template–Neural Cascade for 3D Prediction | 8/9 reviewed leading 3D solutions used template-based modeling directly or used predicted structures as templates for neural refinement; the main alternative was an ensemble of fine-tuned neural predictors. | principles/03.md |
| 4 | Design Positional Information to Extrapolate Beyond Training Lengths | 5/5 top Ribonanza solutions explicitly replaced or avoided fixed absolute position embeddings; length shift was also central in both 3D competitions. | principles/04.md |
| 5 | Model Chains, Copies, and Molecular Context Explicitly | 5/6 reviewed Part 2 solutions explicitly handled chain segmentation, chain matching, per-chain constraints, or chain-aware long-sequence inference; the metric itself permuted equivalent chains. | principles/05.md |
| 6 | Validate on the Same Kind of Novelty the Hidden Set Contains | All 3 competition families exposed major random-split failure modes: sequence duplication, cluster imbalance, temporal leakage, longer hidden RNAs, or newly synthesized targets. | principles/06.md |
| 7 | Use Measurement Uncertainty Without Throwing Away Most Labels | 5/5 top Ribonanza solutions used filtering, sampling, masking, or weighting based on signal-to-noise and per-position errors. | principles/07.md |
| 8 | Ablate Biological Side Information by Regime, Not by Feature Count | All three competitions tested BPP, secondary structure, MSA, language-model embeddings, or external folds, but their value differed sharply by data regime and model. | principles/08.md |
| 9 | Use Pretraining and Curriculum to Match the Real Generalization Gap | Multiple high-ranking solutions in every task family used pretrained RNA encoders, autoencoding/MLM, fine-tuned structure models, or length/quality curricula, but several writeups also reported neutral pretraining—making selection conditional rather than universal. | principles/09.md |
| 10 | Treat Pseudo-Labels as Noisy Measurements, Not Ground Truth | 3/5 top Ribonanza solutions used confidence-filtered pseudo or corrected labels; other strong teams found unfiltered pseudo-labeling neutral or harmful. | principles/10.md |
| 11 | Optimize the Five 3D Outputs as a Portfolio | Every reviewed 3D solution was shaped by best-of-five scoring; top approaches mixed templates, models, seeds, MSA settings, or refinements rather than submitting five near-duplicates. | principles/11.md |
| 12 | Route Models and Compute Per Target | 8/9 reviewed leading 3D solutions used length-, template-, runtime-, or confidence-dependent routing; Part 2 pipelines made this especially explicit under the eight-hour notebook limit. | principles/12.md |
| 13 | Post-Process According to Candidate Provenance and the Metric | All leading TBM pipelines repaired gaps and local geometry, while several neural-model writeups reported that extra coordinate constraints or averaging made predictions worse. | principles/13.md |
| 14 | Ensemble Profile Predictions by Error and Complementarity | Most Ribonanza writeups used multi-model blending, often across architectures, folds, feature sets, targets, or augmentations. | principles/14.md |

## How to Use

Use Glob to list principles/*.md, then Read the principles relevant to your task. The references/ directory holds the raw ranked writeups these principles were summarized from.

Use these principles as a sequence of decisions: identify the output geometry, choose representations and template/neural branches, construct shift-aware validation, then train with quality-aware supervision and allocate inference budget metric-wise. The central theme is calibrated hybridity—combine sequence, pair, template, and neural evidence only where each source is reliable, while preserving diversity against hidden biological and experimental shifts.
