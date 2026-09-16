---
description: >-
  ML playbook for spatial omics cellular imaging competitions. Use when tackling a Kaggle-style competition involving spatial omics cellular imaging. Teaches how to reason about choose the learning unit from the target geometry, preserve channel meaning and biological scale, decompose large fields without throwing away context, make acquisition domain a first-class modeling variable. Six Kaggle competitions comprising 96 indexed solution writeups, with detailed review of 24 leading writeups across Human Protein Atlas image and single-cell classification, Recursion cellular perturbation classification, HuBMAP organ and vasculature segmentation, and UBC-OCEAN whole-slide classification.
---

# Spatial Omics and Cellular Imaging Competition Playbook

These competitions turn microscopy or histopathology into image-level labels, cell-level labels, semantic masks, instance masks, or slide-level diagnoses. Their central challenge is not merely visual recognition: biological scale, acquisition batch, stain or channel semantics, weak supervision, and metric-specific decoding often matter as much as the backbone.

**Source material:** Six Kaggle competitions comprising 96 indexed solution writeups, with detailed review of 24 leading writeups across Human Protein Atlas image and single-cell classification, Recursion cellular perturbation classification, HuBMAP organ and vasculature segmentation, and UBC-OCEAN whole-slide classification.

## Principle Index

| # | Principle | Consensus | File |
|---|-----------|-----------|------|
| 1 | Choose the Learning Unit From the Target Geometry | 6/6 competition families; every successful methodology matched the model's unit of reasoning to the submission unit, although the resulting architectures formed several distinct clusters. | principles/01.md |
| 2 | Preserve Channel Meaning and Biological Scale | 6/6 competition families explicitly managed channels, resolution, magnification, receptive field, or pixel size. | principles/02.md |
| 3 | Decompose Large Fields Without Throwing Away Context | 5/6 families made decomposition central; the remaining perturbation task still aggregated multiple sites rather than relying on a single view. | principles/03.md |
| 4 | Make Acquisition Domain a First-Class Modeling Variable | 6/6 families reported meaningful source variation: stain and institution, HPA versus HuBMAP, plate and experiment, cell line, or WSI. | principles/04.md |
| 5 | Exploit Metadata and Hard Biological Constraints Deliberately | 6/6 families used metadata or task structure, including antibody identity, plate candidate sets, sites, organs, pixel size, WSI origin, glomeruli, or image type. | principles/05.md |
| 6 | Validate on the Acquisition Hierarchy, Not Random Images | 6/6 families emphasized split design; top writeups grouped by WSI, experiment, patient/source, antibody or duplicate identity, and stratified rare labels or organs. | principles/06.md |
| 7 | Bridge Weak Labels With Multi-Instance Consistency | 2/6 families were intrinsically weakly supervised, and leading solutions in both repeatedly used CAM, MIL, dual-level heads, or selected-instance learning; related ideas also appeared in noisy segmentation data. | principles/07.md |
| 8 | Treat Label Quality, External Data, and Pseudo-Labels as One Lifecycle | 6/6 families used at least one of external images, controls, manual relabeling, unannotated images, pseudo-labeling, or mined instances; gains depended strongly on filtering and domain match. | principles/08.md |
| 9 | Optimize Rare Classes and Open-Set Behavior for the Actual Metric | 5/6 families had severe imbalance, rare morphology, or an explicit unseen/outlier class; leading solutions used weighted sampling/losses, targeted mining, metric learning, or synthetic negatives. | principles/09.md |
| 10 | Augment Acquisition Nuisance, Not Biological Identity | 6/6 families used geometric and intensity augmentation, but top solutions tailored it to orientation, channel physics, stain, scale, organ, or domain. | principles/10.md |
| 11 | Prefer Strong Pretrained Representations and Stable Training Over Exotic Heads | 6/6 families relied on pretrained CNNs or transformers; several winners explicitly reported simple heads outperforming more elaborate architectures, while EMA, SWA, GroupNorm, checkpoint averaging, or frozen feature extraction improved stability. | principles/11.md |
| 12 | Decode Predictions With the Metric and Data Constraints in Mind | 6/6 families used consequential post-processing: per-class or per-organ thresholds, structured assignment, confidence refinement, border penalties, size filters, WBF/NMS, outlier rejection, or mask threshold tuning. | principles/12.md |
| 13 | Ensemble Independent Error Sources, Not Redundant Checkpoints | 6/6 families reported ensemble gains, but several also found that excessive TTA or many near-identical models added cost without private-score improvement. | principles/13.md |
| 14 | Design the Submission Pipeline as a Resource-Constrained System | 5/6 families explicitly discussed inference time, memory, image I/O, segmentation speed, or hidden-test robustness; it was especially decisive for WSI and code competitions. | principles/14.md |

## How to Use

Use Glob to list principles/*.md, then Read the principles relevant to your task. The references/ directory holds the raw ranked writeups these principles were summarized from.

Use the principles in sequence: align the learning unit and physical representation first, validate across the true acquisition hierarchy, then add weak-label handling, external data, and metric-aware decoding. In this task family, trustworthy domain-aware validation is the control system that decides whether every later technique is genuinely useful.
