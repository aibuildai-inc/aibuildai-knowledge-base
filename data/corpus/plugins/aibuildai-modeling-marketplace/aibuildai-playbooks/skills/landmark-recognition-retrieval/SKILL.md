---
description: >-
  ML playbook for landmark recognition retrieval competitions. Use when tackling a Kaggle-style competition involving landmark recognition retrieval. Teaches how to reason about select backbone based on pretraining dataset scale, not just architecture, train progressively: clean→noisy data, small→large images, use arcface with dynamic margins based on class frequency, use gem pooling with learned or fixed p≈3. 40+ top-solution writeups across 14 competitions: Google Landmark Recognition/Retrieval 2019-2021, Google Universal Image Embedding, Image Matching Challenge 2023-2025, Hotel-ID, and others
---

# Landmark Recognition & Retrieval Playbook

Landmark recognition and retrieval tasks require learning discriminative visual representations to identify or retrieve specific landmarks, buildings, or instances from large-scale image databases. The core challenge is balancing global context (what landmark is this?) with local details (which exact view/instance?), while handling massive class imbalance, noisy training data, and distinguishing true landmarks from non-landmark images. Success depends on choosing the right embedding approach, progressive training strategy, and sophisticated re-ranking to handle the unique characteristics of instance-level recognition.

**Source material:** 40+ top-solution writeups across 14 competitions: Google Landmark Recognition/Retrieval 2019-2021, Google Universal Image Embedding, Image Matching Challenge 2023-2025, Hotel-ID, and others

## Principle Index

| # | Principle | Consensus | File |
|---|-----------|-----------|------|
| 1 | Select Backbone Based on Pretraining Dataset Scale, Not Just Architecture | Widespread across top solutions 2020-2023. 1st place Universal Embedding explicitly tested this progression. | principles/01.md |
| 2 | Train Progressively: Clean→Noisy Data, Small→Large Images | 5/5 top landmark solutions 2020-2021 used 3-stage training. Recognized as standard approach. | principles/02.md |
| 3 | Use ArcFace with Dynamic Margins Based on Class Frequency | 4/5 top landmark solutions 2020-2021. Introduced by 3rd place 2020, adopted widely after. | principles/03.md |
| 4 | Use GeM Pooling with Learned or Fixed p≈3 | 10+ solutions across all years. Standard pooling choice for landmark tasks. | principles/04.md |
| 5 | Freeze Backbone and Train Head First, Then Unfreeze with Stratified Learning Rates | Universal Embedding 1st/2nd/5th place, multiple landmark solutions. Standard for fine-tuning pretrained models. | principles/05.md |
| 6 | Re-rank Predictions by Penalizing Similarity to Non-Landmarks | 1st place landmark recognition 2020/2021, multiple top retrieval solutions. Critical for GAP metric. | principles/06.md |
| 7 | Aggregate Top-K Neighbors with Power-Weighted Voting, Not Just Top-1 | 3rd place landmark 2020, multiple retrieval solutions. Significant boost over top-1. | principles/07.md |
| 8 | Extend Index Set Beyond Clean Training Data for Retrieval | 1st place landmark 2020 recognition/retrieval, multiple top solutions. Significant gain. | principles/08.md |
| 9 | Ensemble by Concatenating Then Scaling Embeddings, Not Averaging Similarities | 1st place landmark 2020/2021, universal embedding top solutions. Standard ensembling method. | principles/09.md |
| 10 | Apply Test-Time Augmentation with Aspect-Ratio-Aware Resizing | Universal embedding 5th place detailed this explicitly; landmark solutions used multi-scale TTA. | principles/10.md |
| 11 | Use Minimal Augmentation for Metric Learning: Horizontal Flip, Shift-Scale-Rotate, Cutout | 3rd place landmark 2020 published exact augmentation config, adopted by 2021 winners. | principles/11.md |
| 12 | For Image Matching (SfM), Use Sparse Local Features with Match Filtering Over Dense Matchers | Top 3 image matching challenge 2024 all used ALIKED+LightGlue; dense matchers underperformed. | principles/12.md |
| 13 | Validate with Task-Appropriate Data Splits: Use Previous Competition Test Sets When Available | 2021 1st place, 2020 3rd place both used 2019 test set for validation. High correlation with LB. | principles/13.md |
| 14 | Choose Global-Only vs. Local+Global Based on Computational Budget and Marginal Gain | 1st place 2020 used global-only, 1st place 2021 used local+global (DOLG). Mixed approaches across top solutions. | principles/14.md |

## How to Use

Use Glob to list principles/*.md, then Read the principles relevant to your task. The references/ directory holds the raw ranked writeups these principles were distilled from.

These principles work together as a system: strong pretraining provides the foundation, progressive training builds robust features, metric learning with dynamic margins handles class imbalance, and sophisticated re-ranking extracts maximum performance from the embeddings. Start with global descriptors for iteration speed, validate on realistic data, and add complexity (local features, ensembles) only when simpler approaches plateau. The gap between a good solution and a winning one often lies in post-processing and inference details, not just model architecture.
