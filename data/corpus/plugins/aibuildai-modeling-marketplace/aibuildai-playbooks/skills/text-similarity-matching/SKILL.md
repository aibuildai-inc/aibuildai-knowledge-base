---
description: >-
  ML playbook for text similarity matching competitions. Use when tackling a Kaggle-style competition involving text similarity matching. Teaches how to reason about prevent leakage with structure-aware cross-validation, use the other items tied to the same anchor as features, use metric learning (arcface/triplet) for embedding quality, employ two-stage pipeline: retrieval then classification. 33 top-solution writeups across 4 competitions: Shopee Product Matching, Quora Question Pairs, Avito Duplicate Ads Detection, and ICDM Cross-Device Connections
---

# Text Similarity Matching Playbook

Text similarity matching tasks require identifying whether pairs or groups of text (often with images) represent the same entity, question, or product. The core challenge is learning representations that capture semantic similarity while being robust to paraphrasing, multilingual variations, and domain-specific nuances. Success depends on effective embedding learning, strategic multimodal fusion when applicable, and sophisticated post-processing to refine match predictions.

**Source material:** 33 top-solution writeups across 4 competitions: Shopee Product Matching, Quora Question Pairs, Avito Duplicate Ads Detection, and ICDM Cross-Device Connections

## Principle Index

| # | Principle | Consensus | File |
|---|-----------|-----------|------|
| 1 | Prevent Leakage with Structure-Aware Cross-Validation | Source writeups that described their CV split kept related items together (Shopee GroupKFold by label_group, Avito non-overlapping items) | principles/01.md |
| 2 | Use the Other Items Tied to the Same Anchor as Features | Quora solutions built features from the other questions each question was paired with (common neighbors, node degree, component size) | principles/02.md |
| 3 | Use Metric Learning (ArcFace/Triplet) for Embedding Quality | 18/21 Shopee solutions, appearing in other multimodal tasks | principles/03.md |
| 4 | Employ Two-Stage Pipeline: Retrieval Then Classification | 12/21 Shopee solutions, 3rd place Shopee, multiple Quora solutions | principles/04.md |
| 5 | Fuse Multimodal Signals at Multiple Levels | 15/21 Shopee solutions | principles/05.md |
| 6 | Refine Embeddings with Iterative Neighborhood Blending | 1st Shopee, 7th/8th Shopee adopted Query Expansion / DBA variants | principles/06.md |
| 7 | Combine Deep Embeddings with TF-IDF for Diversity | 16/21 Shopee solutions, multiple Quora solutions | principles/07.md |
| 8 | Tune Multiple Thresholds Jointly Across Pipeline Stages | 12/15 solutions discussing thresholds (Shopee 1st, 4th, 6th, 7th) | principles/08.md |
| 9 | Apply Graph-Based Post-Processing to Enforce Consistency | 3rd Shopee (clustering), 2nd Shopee (betweenness), 14th Shopee (clique), Quora transitivity | principles/09.md |
| 10 | Ensemble with Model Diversity, Not Just Architecture Variety | Recurring among solutions discussing ensembling (Quora, Shopee). | principles/10.md |
| 11 | Handle Language/Domain Diversity with Specialized Components | Shopee Indonesian language specialization was a recurring pattern in the source writeups. | principles/11.md |
| 12 | Leverage Domain-Specific Features as Ensemble Signals | Shopee (perceptual hash, image size), Avito (location, price), Quora (qid overlap) | principles/12.md |
| 13 | Weight Samples by Inverse Group Size for Imbalanced Metrics | 2nd Shopee, mentioned in LightGBM discussions | principles/13.md |
| 14 | Optimize for the Actual Metric, Not Proxy Losses | Shopee solutions tuned match thresholds directly on F1; Quora solutions rescaled predictions or resampled training data to the lower duplicate rate of the test set for log loss | principles/14.md |

## How to Use

Use Glob to list principles/*.md, then Read the principles relevant to your task. The references/ directory holds the raw ranked writeups these principles were distilled from.

Success in text similarity matching comes from the interaction of strong embeddings, smart multimodal fusion, and aggressive post-processing. Treat embeddings as a starting point, not the solution—the magic is in how you retrieve, refine, and fuse them. Always validate that CV correlates with LB by using structure-aware splits.
