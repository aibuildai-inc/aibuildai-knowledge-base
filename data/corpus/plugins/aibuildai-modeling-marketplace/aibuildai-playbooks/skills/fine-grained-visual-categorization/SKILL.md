---
description: >-
  ML playbook for fine grained visual categorization competitions. Use when tackling a Kaggle-style competition involving fine grained visual categorization. Teaches how to reason about choose architecture based on task type: individual id vs species classification, apply sub-center arcface with dynamic margins for long-tailed individual id, localize the region of interest with bounding box detection before classification, handle long-tail distribution with staged training: frequent classes first, then all classes. Analysis of 44 top-solution writeups across 8 competitions including whale identification, flower classification, fashion and furniture categorization, and wildlife monitoring.
---

# Fine-Grained Visual Categorization Playbook

Fine-grained visual categorization tasks require distinguishing between visually similar categories (plant species, animal individuals, product types) where differences are often subtle. The core challenge is extracting discriminative features at a finer granularity than standard classification, often compounded by extreme class imbalance, long-tailed distributions, and the need to identify novel instances not seen during training.

**Source material:** Analysis of 44 top-solution writeups across 8 competitions including whale identification, flower classification, fashion and furniture categorization, and wildlife monitoring.

## Principle Index

| # | Principle | Consensus | File |
|---|-----------|-----------|------|
| 1 | Choose Architecture Based on Task Type: Individual ID vs Species Classification | Both individual identification competitions (Happy Whale, Humpback) are dominated by metric learning (ArcFace, CosFace, triplet, or siamese losses in 22 of their 29 writeups); species and category tasks mostly use softmax. | principles/01.md |
| 2 | Apply Sub-Center ArcFace with Dynamic Margins for Long-Tailed Individual ID | 1st place Happy Whale used sub-center ArcFace with class-frequency-based dynamic margins; 4th, 6th, 10th, and 36th place Happy Whale also used dynamic margins. | principles/02.md |
| 3 | Localize the Region of Interest with Bounding Box Detection Before Classification | Top whale/dolphin solutions (1st and 2nd place Happy Whale with YOLOv5/YOLOX, 3rd place Humpback) crop subjects with custom-trained detectors, and all 5 iWildCam 2021 writeups crop animals with MegaDetector. | principles/03.md |
| 4 | Handle Long-Tail Distribution with Staged Training: Frequent Classes First, Then All Classes | 1st place Humpback trained first on classes with more than 10 samples, then on all samples with most of the network frozen. | principles/04.md |
| 5 | Leverage Domain-Specific External Data and Pretraining Over Generic ImageNet | Mixed evidence - iWildCam 2021 (3rd) added iNaturalist images for shared classes and Flower Classification (1st) added external flower datasets, but Flower (1st) found iNaturalist and OpenImages data lowered accuracy; validate each external source | principles/05.md |
| 6 | Apply Pseudo-Labeling Iteratively with Confidence Filtering for Test Set Leverage | 1st-3rd place Happy Whale, 1st place Humpback, 2nd place Happy Whale all use 2-5 rounds of pseudo-labeling with thresholds 0.4-0.8. | principles/06.md |
| 7 | Ensemble Diverse Models: Different Backbones, Resolutions, and Pretraining Sources | 1st place Happy Whale ensembled around 50 models; most other top solutions also ensembled models with different backbones or input resolutions. | principles/07.md |
| 8 | Use GeM Pooling Instead of Global Average Pooling to Emphasize Salient Features | 1st place Happy Whale, 3rd place Happy Whale use GeM pooling with p=3. | principles/08.md |
| 9 | Exploit Dataset Constraints with Post-Processing Rules | 1st place Humpback rebalanced its top-5 predictions toward classes unused as top-1 (+0.001 to 0.002); constraint- and balance-based post-processing can give small gains. | principles/09.md |
| 10 | Apply Multi-Scale Test-Time Augmentation with Horizontal Flipping and Multi-Crop | 9 of the 44 writeups mention TTA; typical setup is a horizontal flip plus a few crops or slight rotations. | principles/10.md |
| 11 | Combine k-NN and Logit Predictions for Metric Learning to Balance Precision and Recall | 1st place Happy Whale mixed k-NN and logits with ratio 0.5 (no pseudo-labels) to 0.8 (after pseudo-labels); 3rd place used similar mixing. | principles/11.md |
| 12 | Set Learning Rate for Classification Head 5-10x Higher Than Backbone | 1st place Happy Whale set the ArcFace head learning rate 10x the backbone learning rate and reported a significant gain. | principles/12.md |

## How to Use

Use Glob to list principles/*.md, then Read the principles relevant to your task. The references/ directory holds the raw ranked writeups these principles were distilled from.

These principles form a decision sequence: start by identifying the task type (individual ID vs classification) to choose the core architecture, then address data quality (localization), handle class imbalance (staged training, dynamic margins), leverage external knowledge (domain pretraining, pseudo-labeling), and finally optimize for maximum performance (ensembling, TTA, post-processing). Each principle gates subsequent ones—choosing metric learning changes your pooling, loss, and inference strategy. Apply them iteratively: validate each decision on a held-out set before moving to the next.
