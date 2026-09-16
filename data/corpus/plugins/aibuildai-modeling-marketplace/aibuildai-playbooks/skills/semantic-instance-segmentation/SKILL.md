---
description: >-
  ML playbook for semantic instance segmentation competitions. Use when tackling a Kaggle-style competition involving semantic instance segmentation. Teaches how to reason about choose architecture family by problem complexity, add dilated convolutions to bottleneck for u-net architectures, use multi-scale training and inference as default, combine bce and dice loss for mask prediction. 26 top-solution writeups across 5 competitions: Carvana (binary car masking), iMaterialist Fashion 2019/2020 (fine-grained fashion with attributes), Open Images 2019 (300-class hierarchical dataset), and NFL Helmet Assignment (detection + tracking + assignment)
---

# Semantic Instance Segmentation Playbook

Semantic instance segmentation requires predicting pixel-precise masks for each individual object instance in an image, often across multiple classes. The core challenge is balancing spatial precision (accurate mask boundaries), instance discrimination (separating overlapping objects), and class prediction - all while handling extreme scale variation, class imbalance, and computational constraints. Solutions split into encoder-decoder architectures (U-Net family) for simpler domains and detection-based two-stage approaches (Mask R-CNN family) for complex multi-class scenarios.

**Source material:** 26 top-solution writeups across 5 competitions: Carvana (binary car masking), iMaterialist Fashion 2019/2020 (fine-grained fashion with attributes), Open Images 2019 (300-class hierarchical dataset), and NFL Helmet Assignment (detection + tracking + assignment)

## Principle Index

| # | Principle | Consensus | File |
|---|-----------|-----------|------|
| 1 | Choose Architecture Family by Problem Complexity | 5/5 competitions show clear architecture clustering by task type | principles/01.md |
| 2 | Add Dilated Convolutions to Bottleneck for U-Net Architectures | 3/5 Carvana top solutions used dilated convolutions; 3rd place specifically ablated +0.0013 improvement | principles/02.md |
| 3 | Use Multi-Scale Training and Inference as Default | 5/5 competitions - all top-3 solutions used multi-scale | principles/03.md |
| 4 | Combine BCE and Dice Loss for Mask Prediction | 4/5 top Carvana solutions used BCE+Dice; Fashion winners used focal loss (BCE variant) | principles/04.md |
| 5 | Apply Conservative Augmentation - Strong Transforms Often Hurt | 4/5 competitions - multiple solutions explicitly reported worse scores with aggressive augmentation | principles/05.md |
| 6 | Tune Post-Processing Thresholds on Validation Data | 5/5 competitions - every top solution tuned confidence/NMS/mask thresholds, often +0.01-0.03 gain | principles/06.md |
| 7 | Rebalance Training Data for Severe Class Imbalance | 2/2 large-scale multi-class competitions (Open Images, Fashion) - top solutions rebalanced | principles/07.md |
| 8 | Ensemble with Test-Time Augmentation at Multiple Stages | 5/5 competitions - all top-3 solutions used TTA, often at proposal/bbox/mask levels separately | principles/08.md |
| 9 | Use Hierarchical or Second-Level Models for Complex Attributes | 3/3 Fashion/Open Images solutions addressed hierarchy/attributes with separate strategies | principles/09.md |
| 10 | Train at Lower Resolution, Fine-Tune at Target Resolution | 3/5 competitions - multiple solutions used progressive resolution (Fashion 2nd: 800×1266 → 1024×1600) | principles/10.md |
| 11 | Implement Domain-Specific Post-Processing for Systematic Errors | 3/5 competitions - Carvana (antenna merging), Fashion (mask overlap removal), NFL (duplicate label handling) | principles/11.md |
| 12 | Leverage Pseudo-Labeling When Test Set Is Large | 2/3 simple segmentation tasks (Carvana) where test set ~= train set size | principles/12.md |
| 13 | Build Ensemble Diversity Through Architecture and Scale, Not Just Seeds | 5/5 competitions - top ensembles used different backbones/architectures, not just different random seeds | principles/13.md |
| 14 | Use Pretrained Weights and Fine-Tune All Layers | 5/5 competitions - every top solution started from COCO or ImageNet pretrained weights | principles/14.md |
| 15 | Validate with Stratified Splits and Monitor CV-LB Correlation | 4/5 competitions - solutions explicitly mentioned CV strategy and tracked CV-LB gap | principles/15.md |

## How to Use

Use Glob to list principles/*.md, then Read the principles relevant to your task. The references/ directory holds the raw ranked writeups these principles were distilled from.

Apply these principles in sequence: choose architecture based on task complexity, then build the training pipeline (multi-scale, conservative augmentation, rebalancing, combined loss), tune post-processing thresholds, and finally ensemble diverse models with TTA. The biggest gains come from architecture choice, multi-scale inference, and threshold tuning - together these account for 60-80% of the gap between a baseline and a winning solution.
