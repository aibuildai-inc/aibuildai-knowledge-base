---
description: >-
  ML playbook for recommendation engagement competitions. Use when tackling a Kaggle-style competition involving recommendation engagement. Teaches how to reason about choose base model architecture based on feature density, apply negative downsampling for extreme imbalance (>99:1), engineer multi-level aggregations from related tables, create conditional probability features for categorical interactions. 50+ top-solution writeups across 30 competitions including Home Credit Default Risk, Avazu CTR, TalkingData Ad Fraud, KKBox Music Recommendation, Outbrain Click Prediction, and Santander transaction prediction.
---

# Recommendation & Engagement Prediction Playbook

Recommendation and engagement prediction tasks involve predicting user actions (clicks, conversions, churn, transactions) based on user-item interactions, often with temporal dynamics and extreme class imbalance. The core challenge is extracting signal from high-cardinality categorical features (millions of users/items) while handling severe data sparsity, temporal drift, and the cold-start problem for new users/items.

**Source material:** 50+ top-solution writeups across 30 competitions including Home Credit Default Risk, Avazu CTR, TalkingData Ad Fraud, KKBox Music Recommendation, Outbrain Click Prediction, and Santander transaction prediction.

## Principle Index

| # | Principle | Consensus | File |
|---|-----------|-----------|------|
| 1 | Choose Base Model Architecture Based on Feature Density | Majority of CTR/click competitions (Avazu #1, Criteo #1, Outbrain #3/#4). Credit/transaction competitions favor GBMs (Home Credit top 10, Santander top 10). | principles/01.md |
| 2 | Apply Negative Downsampling for Extreme Imbalance (>99:1) | TalkingData #1 (99.8% negatives downsampled 1:1), multiple ad fraud/CTR solutions. Not used in balanced problems like Home Credit. | principles/02.md |
| 3 | Engineer Multi-Level Aggregations from Related Tables | Home Credit top 5 all created 500-3000+ aggregated features. KKBox #1 used 400 features. Standard in all relational datasets. | principles/03.md |
| 4 | Create Conditional Probability Features for Categorical Interactions | KKBox #1 used P(source_type|user), P(artist|user) as top features. Multiple CTR solutions used P(click|user,ad) style features. | principles/04.md |
| 5 | Extract Time Delta and Lagged Target Features | TalkingData #5 (time_diff forward/backward as top features), KKBox #1 (timestamp index), Home Credit #1 (lag features from last 5 applications). | principles/05.md |
| 6 | Embed High-Cardinality Categories with Matrix Factorization | TalkingData #1 (LDA/NMF/PCA on user-app matrix), KKBox #1 (SVD on user-song and user-artist matrices), used in 5+ top solutions. | principles/06.md |
| 7 | Use Time-Based Train/Validation Split for Temporal Data | KKBox #1 (last 20% by time), TalkingData #1 (day 7-8 train, day 9 val), standard across temporal competitions. | principles/07.md |
| 8 | Detect and Exploit Train/Test Distribution Differences | Santander #1 (identified real vs. fake test samples to enable >0.92 LB), Home Credit #1 (0.98 AUC train/test classifier), Outbrain split present/future test. | principles/08.md |
| 9 | Ensemble GBM and Neural Networks for Complementary Patterns | KKBox #1 (0.6 LGBM + 0.4 NN), Home Credit #1 (GBM + NN stacking), TalkingData #1 (7 GBM + 1 NN blend). Standard 2-model diversity approach. | principles/09.md |
| 10 | Use Field-Aware Neural Network Architectures for Multi-Entity Data | KKBox #1 (separate user/song/context fields with independent embeddings then concatenate), standard in recommendation NNs. | principles/10.md |
| 11 | Stack Multiple Levels with Raw Feature Restacking | Home Credit #1 (3-level stacking with raw features at L2), standard in top solutions with large teams. | principles/11.md |
| 12 | Engineer Uniqueness and Count-Based Features | Santander #1 (uniqueness features were the 'magic' enabling 0.927 LB), TalkingData top solutions (nunique counts, top-k value ratios). | principles/12.md |
| 13 | Create Nested Models on Sub-Tables for Implicit Target Propagation | Home Credit #5 (nested models on 6 sub-tables, +0.002 CV), unique to multi-table credit risk tasks. | principles/13.md |
| 14 | Use Target Encoding with Strict Time-Based or Fold-Based Holdout | Standard across competitions, but TalkingData #5 (lagged target encoding by day), Home Credit top solutions (careful CV to avoid leakage). | principles/14.md |
| 15 | Validate Model Selection Against Business Metrics, Not Just AUC | KKBox #1 raised the question of AUC vs. user-level metrics (GAUC). Implicit in most engagement tasks. | principles/15.md |

## How to Use

Use Glob to list principles/*.md, then Read the principles relevant to your task. The references/ directory holds the raw ranked writeups these principles were distilled from.

These principles compose into a full pipeline: start with model selection (FFM vs. GBM based on sparsity), handle imbalance if needed, engineer aggregations and conditional probabilities, embed high-cardinality IDs, encode targets carefully with holdout, split by time, check for distribution shifts, ensemble GBM+NN, and stack if you have the models. Prioritize validation hygiene—leakage destroys generalization.
