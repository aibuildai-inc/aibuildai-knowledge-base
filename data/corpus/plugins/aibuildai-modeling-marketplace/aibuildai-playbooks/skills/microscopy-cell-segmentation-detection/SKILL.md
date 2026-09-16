---
description: >-
  ML playbook for microscopy cell segmentation detection competitions. Use when tackling a Kaggle-style competition involving microscopy cell segmentation detection. Teaches how to reason about choose the prediction representation from the output geometry, select 2d, 2.5d, or 3d from anisotropy, context, and compute, set resolution, output stride, and target width in physical units, encode ownership cues when foreground masks cannot separate instances. 33 top-ranked solution writeups reviewed across 5 directly relevant competitions: CZII CryoET Object Identification, Sartorius Cell Instance Segmentation, Data Science Bowl 2018, BYU Flagellar Motor Localization, and SenNet/HOA Blood Vessel Segmentation. Image Matching Challenge 2025 and MABe Mouse Behavior Detection were excluded because their objectives and data are not microscopy segmentation or detection.
---

# Microscopy Cell Segmentation and Detection Playbook

Microscopy competitions span several output geometries: individual 2D masks, dense 3D structures, and point locations in noisy tomograms. The central challenge is to match the representation, physical resolution, validation design, and decoder to the metric while learning from very few biological specimens and surviving acquisition-domain shift.

**Source material:** 33 top-ranked solution writeups reviewed across 5 directly relevant competitions: CZII CryoET Object Identification, Sartorius Cell Instance Segmentation, Data Science Bowl 2018, BYU Flagellar Motor Localization, and SenNet/HOA Blood Vessel Segmentation. Image Matching Challenge 2025 and MABe Mouse Behavior Detection were excluded because their objectives and data are not microscopy segmentation or detection.

## Principle Index

| # | Principle | Consensus | File |
|---|-----------|-----------|------|
| 1 | Choose the Prediction Representation from the Output Geometry | 5/5 relevant competitions; every top tier first converted the official target into a representation suited to its geometry rather than treating all problems as generic segmentation. | principles/01.md |
| 2 | Select 2D, 2.5D, or 3D from Anisotropy, Context, and Compute | 3/3 volumetric competitions explicitly compared dimensionality; successful solutions ranged from full 3D to multiview 2D and 2.5D, so there is no universal winner. | principles/02.md |
| 3 | Set Resolution, Output Stride, and Target Width in Physical Units | 5/5 competitions; top solutions repeatedly gained from larger cell crops, voxel-spacing normalization, class-specific radii, or deliberately coarse heatmaps matched to metric tolerance. | principles/03.md |
| 4 | Encode Ownership Cues When Foreground Masks Cannot Separate Instances | 2/2 instance-mask competitions; the strongest alternative clusters were box/ROI masks and center-flow/watershed systems, with point-localization competitions independently validating heatmap centers. | principles/04.md |
| 5 | Validate by Biological Acquisition Unit and Domain | 5/5 competitions; experiment-level, donor-level, cell-type/modality, voxel-size, or source-dataset splits were necessary, although tiny numbers of specimens made absolute CV noisy. | principles/05.md |
| 6 | Treat Annotation Quality and Domain Match as Model Components | 4/5 competitions benefited materially from corrected labels, carefully staged pseudo-labels, or aligned external data; CZII top teams often found synthetic or supplemental data neutral or harmful, making the condition itself important. | principles/06.md |
| 7 | Control Patch Sampling Before Increasing Model Capacity | 5/5 competitions used some combination of positive crops, class-balanced sampling, small-object anchors, non-empty sampling, hard negatives, or crop-scale control. | principles/07.md |
| 8 | Match Targets and Loss Weighting to Imbalance and Metric Geometry | 5/5 competitions customized target radius, positive/class weights, boundary terms, hard-voxel mining, focal/Tversky variants, or auxiliary heads; simple BCE/CE also won when the target representation was well chosen. | principles/08.md |
| 9 | Augment Acquisition Nuisances While Preserving Biological Semantics | 5/5 competitions relied heavily on augmentation, especially with few labeled specimens; geometric symmetries, scale variation, intensity shifts, blur/noise, and MixUp/CutMix appeared repeatedly. | principles/09.md |
| 10 | Stitch Tiles with Edge-Aware Weighting and Recover Missing Context with Views | 5/5 competitions used large crops, tiled inference, multiscale inference, or TTA; all 3 volumetric competitions explicitly addressed sliding-window edges or multiview aggregation. | principles/10.md |
| 11 | Decode Probabilities with the Topology Assumed by the Task | 5/5 competitions used task-specific decoding: connected components or local maxima for particles, watershed/flows or mask NMS for cells, and connectivity/morphology constraints for vessels. | principles/11.md |
| 12 | Calibrate the Operating Point on the Exact Metric | 5/5 competitions treated thresholds, class-specific filters, mask size, or confidence reranking as major parameters rather than afterthoughts. | principles/12.md |
| 13 | Ensemble at the Earliest Compatible Representation | 4/5 competitions had top-tier probability, box, mask, fold, seed, or architecture ensembles; single models also placed first or near first in DSB 2018, BYU, and SenNet, showing that ensembling is conditional rather than mandatory. | principles/13.md |

## How to Use

Use Glob to list principles/*.md, then Read the principles relevant to your task. The references/ directory holds the raw ranked writeups these principles were summarized from.

Use the principles as a sequence: infer target topology and physical scale first, then choose dimensionality and supervision, build leakage-free evidence, and only afterward tune decoding, thresholds, and ensembles. In this task family, representation, data quality, resolution, and metric-compatible post-processing usually matter more than replacing one strong backbone with another.
