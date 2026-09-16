---
description: >-
  ML playbook for satellite imagery analysis competitions. Use when tackling a Kaggle-style competition involving satellite imagery analysis. Teaches how to reason about choose architecture based on task type and data resolution, handle multi-spectral bands by deciding whether to use them all or filter by task, detect and handle spatial overlaps to avoid data leakage, use two-stage pipeline for severely imbalanced segmentation: classifier first, then segmenter. Analysis of 29 top-solution writeups (ranks 1-58) across 5 satellite imagery competitions: cloud segmentation, ship detection, Amazon deforestation, DSTL multi-class segmentation, and chronology tasks.
---

# Satellite Imagery Analysis Playbook

Satellite imagery analysis tasks involve detecting and segmenting objects or phenomena from overhead imagery captured by satellites or aircraft. These competitions span segmentation (ships, buildings, clouds), classification (land use types), and multi-class detection. The core challenge is handling high-resolution, multi-spectral data with severe class imbalance, spatial/temporal overlaps, and label alignment issues that break standard computer vision assumptions.

**Source material:** Analysis of 29 top-solution writeups (ranks 1-58) across 5 satellite imagery competitions: cloud segmentation, ship detection, Amazon deforestation, DSTL multi-class segmentation, and chronology tasks.

## Principle Index

| # | Principle | Consensus | File |
|---|-----------|-----------|------|
| 1 | Choose Architecture Based on Task Type and Data Resolution | Most top segmentation solutions used U-Net-family models; only 2 of the 6 ship-detection solutions used Mask R-CNN successfully | principles/01.md |
| 2 | Handle Multi-Spectral Bands by Deciding Whether to Use Them All or Filter by Task | Most solutions in the multi-band competitions (DSTL, Planet Amazon) chose bands per class or per model rather than always using all bands | principles/02.md |
| 3 | Detect and Handle Spatial Overlaps to Avoid Data Leakage | Top solutions in both competitions with overlapping images (Airbus ships, Draper chronology) explicitly found and used the overlaps | principles/03.md |
| 4 | Use Two-Stage Pipeline for Severely Imbalanced Segmentation: Classifier First, Then Segmenter | Most top ship-detection solutions and many top cloud solutions used separate empty-image classifiers | principles/04.md |
| 5 | Combine BCE and Dice Loss to Balance Pixel-Wise Accuracy and Region Overlap | BCE+Dice combinations were the most common segmentation loss; pure Dice rarely won | principles/05.md |
| 6 | Apply Test-Time Augmentation with Metric-Compatible Averaging | Most solutions used TTA; reported gains were small but consistent | principles/06.md |
| 7 | Optimize Threshold Separately from Model Training Using Validation Data | Many solutions tuned thresholds explicitly on validation, often together with a minimum mask size | principles/07.md |
| 8 | Use Cross-Validation with Spatial or Temporal Splits to Match Test Distribution | Most solutions used K-fold CV; two top ship-detection solutions built leak-free folds by grouping overlapping images | principles/08.md |
| 9 | Ensemble Diverse Models with Low Correlation to Maximize Gains | Most top solutions ensembled several diverse models | principles/09.md |
| 10 | Exploit Metadata Features When Available, Especially for Group-Based Patterns | Top chronology solutions grouped images by location and by acquisition day before solving each group | principles/10.md |
| 11 | Apply Post-Processing Based on Domain Constraints and Metric Characteristics | Many solutions used post-processing beyond simple thresholding; common: size filtering, morphology, metric-aware pruning | principles/11.md |
| 12 | Understand and Optimize for the Metric's Specific Behavior, Not Just Training Loss | Several solutions explicitly discussed metric vs loss misalignment and tuned decisions separately for the metric | principles/12.md |

## How to Use

Use Glob to list principles/*.md, then Read the principles relevant to your task. The references/ directory holds the raw ranked writeups these principles were distilled from.

These principles work together: start with architecture and data decisions (U-Net, band selection, overlap handling), then optimize training (BCE+Dice loss), and finally tune inference (TTA, threshold, ensemble, post-processing) for the specific metric. Trust local CV when it's properly constructed (spatial folds, metric-aligned validation), but always verify assumptions by visualizing predictions and checking for distribution shifts.
