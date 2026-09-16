---
description: >-
  ML playbook for medical image segmentation competitions. Use when tackling a Kaggle-style competition involving medical image segmentation. Teaches how to reason about choose architecture dimensionality based on data structure and test constraints, match train resolution to test physical scale, not pixel count, use pseudo-labeling to refine sparse annotations or bridge domain gaps, select loss functions that align with the evaluation metric. Analysis of 86 top-solution writeups across 7 medical imaging competitions: blood vessel/vasculature segmentation (3D HiP-CT), organ segmentation (multi-organ histology), cryo-electron tomography object identification, nuclei detection (diverse microscopy), cell instance segmentation, and pneumothorax detection (X-ray).
---

# Medical Image Segmentation Playbook

Medical image segmentation tasks require identifying and delineating anatomical structures or pathological regions in medical imagery. The core challenge is handling extreme variability in imaging protocols, resolutions, staining techniques, and annotation quality while achieving pixel-perfect boundaries that generalize across acquisition methods and patient populations.

**Source material:** Analysis of 86 top-solution writeups across 7 medical imaging competitions: blood vessel/vasculature segmentation (3D HiP-CT), organ segmentation (multi-organ histology), cryo-electron tomography object identification, nuclei detection (diverse microscopy), cell instance segmentation, and pneumothorax detection (X-ray).

## Principle Index

| # | Principle | Consensus | File |
|---|-----------|-----------|------|
| 1 | Choose Architecture Dimensionality Based on Data Structure and Test Constraints | In the 3D blood vessel competition the 1st place used 2.5D models after 3D models did not help; cryo-ET object identification solutions used 3D models | principles/01.md |
| 2 | Match Train Resolution to Test Physical Scale, Not Pixel Count | The competitions whose test resolution differed from training (blood vessel, organ segmentation) emphasize this | principles/02.md |
| 3 | Use Pseudo-Labeling to Refine Sparse Annotations or Bridge Domain Gaps | 5/7 competitions, especially those with sparse/mixed-quality labels | principles/03.md |
| 4 | Select Loss Functions That Align With the Evaluation Metric | Most source competitions; combinations dominate single losses | principles/04.md |
| 5 | Apply Domain-Specific Stain/Intensity Normalization to Bridge Protocol Gaps | Competitions with multi-protocol data (HPA vs HuBMAP, different stains) | principles/05.md |
| 6 | Validate on Splits That Match Test Distribution, Not Random Folds | Most source competitions; random folds caused overfitting in several cases | principles/06.md |
| 7 | Ensemble Multi-Scale TTA and Multi-Axis Predictions for Robust Inference | 7/7 competitions use TTA; multi-axis inference recurs in the 3D blood vessel competition | principles/07.md |
| 8 | Optimize Thresholds Per-Class and Per-Metric Using Validation Data | Most source competitions with multi-class or complex metrics | principles/08.md |
| 9 | Use Instance Segmentation When Objects Must Be Individually Counted or Tracked | 3/3 instance segmentation competitions (cells, nuclei) | principles/09.md |
| 10 | Train on Full Data Without Folds for Final Submission If Convergence Is Stable | Several source competitions; especially those with <10 volumes/patients | principles/10.md |
| 11 | Post-Process Predictions With Connected Components and Morphological Operations | 7/7 competitions use post-processing; effect varies by metric | principles/11.md |
| 12 | Increase Effective Resolution With Interpolation or Intelligent Cropping | Several high-resolution competitions (blood vessels, organs) | principles/12.md |

## How to Use

Use Glob to list principles/*.md, then Read the principles relevant to your task. The references/ directory holds the raw ranked writeups these principles were distilled from.

These principles interact: stain normalization + heavy intensity augmentation address domain shift, while multi-axis TTA + multi-scale inference exploit medical data's orientation/scale invariance. Start with validation strategy and loss function aligned to the metric, then layer in domain-specific techniques (2.5D vs 3D, resolution matching, pseudo-labeling) based on your data characteristics.
