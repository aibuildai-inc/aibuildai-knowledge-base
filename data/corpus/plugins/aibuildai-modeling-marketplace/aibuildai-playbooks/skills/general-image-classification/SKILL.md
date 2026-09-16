---
description: >-
  ML playbook for general image classification competitions. Use when tackling a Kaggle-style competition involving general image classification. Teaches how to reason about choose architecture ensembles for complementary inductive biases, scale batch size aggressively for large noisy datasets, apply augmentation progressively from none to soft to hard, leverage temporal or spatial correlations with nearest neighbor aggregation. 17 top-solution writeups across 5 competitions: QuickDraw Doodles (8 writeups, 340 classes, 50M samples), Inclusive Images (4 writeups, multi-label with 7K classes), State Farm Driver Detection (2 writeups, temporal sequences), Facial Expression (2 writeups, 7 classes), and CIFAR-10 (1 writeup, baseline).
---

# General Image Classification Playbook

General image classification encompasses tasks where you must assign one or more labels to images from a broad set of categories. These tasks span diverse domains: facial expressions, sketch recognition, geographic diversity in labeling, and driver behavior detection. The core challenge is building models that generalize across varied visual domains while handling issues like class imbalance, dataset scale, train-test distribution shifts, and leveraging all available information in the data.

**Source material:** 17 top-solution writeups across 5 competitions: QuickDraw Doodles (8 writeups, 340 classes, 50M samples), Inclusive Images (4 writeups, multi-label with 7K classes), State Farm Driver Detection (2 writeups, temporal sequences), Facial Expression (2 writeups, 7 classes), and CIFAR-10 (1 writeup, baseline).

## Principle Index

| # | Principle | Consensus | File |
|---|-----------|-----------|------|
| 1 | Choose Architecture Ensembles for Complementary Inductive Biases | 5/5 QuickDraw top solutions, 4/4 Inclusive Images solutions used diverse architectures | principles/01.md |
| 2 | Scale Batch Size Aggressively for Large Noisy Datasets | 4/8 QuickDraw solutions explicitly used large batches (400-10K) | principles/02.md |
| 3 | Apply Augmentation Progressively from None to Soft to Hard | 2/4 Inclusive Images solutions explicitly described progressive augmentation | principles/03.md |
| 4 | Leverage Temporal or Spatial Correlations with Nearest Neighbor Aggregation | 2/2 State Farm top solutions used nearest neighbor features | principles/04.md |
| 5 | Tune Class-Specific Thresholds for Multi-Label Tasks | 3/4 Inclusive Images solutions tuned per-class thresholds; winner gained ~0.15 | principles/05.md |
| 6 | Add Domain-Specific Auxiliary Features When Core Signal Is Weak | QuickDraw solutions added non-visual stroke information: the 1st place added time features (the most significant was the maximum stroke timestamp), and other solutions encoded stroke order, speed, or stroke count as image channels | principles/06.md |
| 7 | Apply Test-Time Augmentation with Geometric Averaging for Robustness | 5/8 QuickDraw solutions used TTA | principles/07.md |
| 8 | Weight Loss or Sample to Address Severe Class Imbalance | 2/4 Inclusive Images solutions, 1/8 QuickDraw solutions used class weighting | principles/08.md |
| 9 | Ensemble via Weighted Averaging by Validation Performance Raised to a Power | 3/8 QuickDraw solutions used powered weighting | principles/09.md |
| 10 | Exploit Known Test Set Structure with Post-Processing Constraints | 1/8 QuickDraw solutions (winner); "secret sauce" gave +0.7% | principles/10.md |
| 11 | Average Multiple Checkpoints or Snapshots for Stability | 2/8 QuickDraw solutions averaged checkpoints | principles/11.md |

## How to Use

Use Glob to list principles/*.md, then Read the principles relevant to your task. The references/ directory holds the raw ranked writeups these principles were distilled from.

These principles interact: architecture diversity enables strong ensembles, threshold tuning rescues imbalanced predictions. Start with a strong single model (principles 1-4), then layer in thresholds, ensembling and post-processing (5-11) for the final boost.
