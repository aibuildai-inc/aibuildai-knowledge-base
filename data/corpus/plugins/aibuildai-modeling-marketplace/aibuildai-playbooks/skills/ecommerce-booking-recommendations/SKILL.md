---
description: >-
  ML playbook for ecommerce booking recommendations competitions. Use when tackling a Kaggle-style competition involving ecommerce booking recommendations. Teaches how to reason about adopt a two-stage candidate-rerank architecture, build multiple co-visitation matrices with varied parameters, apply strong recency and trending weights to items, create user-item interaction features across multiple time windows. 27 top-solution writeups across 7 competitions (OTTO Recommender: 20 writeups, plus Expedia Hotels, Airbnb Destinations, and Santander Products)
---

# E-commerce Booking Recommendations Playbook

E-commerce booking recommendations predict which products, destinations, or services users will purchase next based on historical interactions and user/item attributes. The core challenge is handling massive item catalogs (thousands to millions of items), severe class imbalance (users interact with <1% of items), and strong temporal dynamics where recent behavior and trending items dominate relevance. Success requires generating diverse candidates efficiently, then ranking them with rich interaction features.

**Source material:** 27 top-solution writeups across 7 competitions (OTTO Recommender: 20 writeups, plus Expedia Hotels, Airbnb Destinations, and Santander Products)

## Principle Index

| # | Principle | Consensus | File |
|---|-----------|-----------|------|
| 1 | Adopt a Two-Stage Candidate-Rerank Architecture | All top solutions across all 7 competitions use this pattern | principles/01.md |
| 2 | Build Multiple Co-Visitation Matrices with Varied Parameters | Top-3 OTTO solution used 20 co-visitation matrices; universal across competitions | principles/02.md |
| 3 | Apply Strong Recency and Trending Weights to Items | Several top OTTO solutions weight a session's history and its co-visitation counts by recency and action type; one added weekly count trends as features, and another found that trending-item derivatives did not help. | principles/03.md |
| 4 | Create User-Item Interaction Features Across Multiple Time Windows | Several top OTTO solutions built session-item interaction features (counts per action type, time since the last action, last action type) and item counts over several recent windows (11th: 1-day and 3-day windows; 7th: last 1-7 days and last 1-4 weeks); another turned co-visitation counts into one interaction feature per candidate pair (3rd). | principles/04.md |
| 5 | Apply Aggressive Negative Sampling Based on Target Imbalance | All GBDT solutions use negative sampling; ratios vary by target (clicks: 5%, carts: 25%, orders: 40%) | principles/05.md |
| 6 | Use BPR or Matrix Factorization for User-Item Similarity Features | Many top OTTO solutions use BPR, ALS, or related matrix-factorization embeddings as reranker features | principles/06.md |
| 7 | Validate with Time-Based Splits and Measure CV-LB Correlation Early | Universal: all solutions use last week as validation; top OTTO solutions report near-perfect, stable CV-LB correlation | principles/07.md |
| 8 | Include Candidate-Source Features as Ranking Signals | Multiple top OTTO solutions tag candidates with their generating retrieval method and use it as a ranking feature | principles/08.md |
| 9 | Tune Candidate Count to Balance Recall and Computation | Top solutions generate 100-500 candidates; OTTO 1st uses 1200; 7th OTTO improves LB 0.595→0.598 by increasing 384→1024 | principles/09.md |
| 10 | Handle Cold-Start Users with Popularity Fallbacks | Top OTTO solutions build dedicated candidate sources for cold-start sessions (first-interaction pairing) and fall back to popularity-based candidates | principles/10.md |
| 11 | Optimize Ensemble Weights on Validation, Not Leaderboard | Most top OTTO solutions ensemble multiple models with equal or validation-tuned weights; OTTO 7th place ensembles LightGBM + GNN | principles/11.md |
| 12 | Differentiate Reranker Objectives by Target Type | OTTO winners train separate models for clicks/carts/orders | principles/12.md |

## How to Use

Use Glob to list principles/*.md, then Read the principles relevant to your task. The references/ directory holds the raw ranked writeups these principles were distilled from.

These principles work together as a pipeline: start with diverse candidate generation (co-visitation + popularity + revisits) to maximize recall, then rerank with rich interaction and similarity features to achieve final precision. Temporal dynamics (recency, action type) are critical in session-based recommendation; optimize them first before deep modeling.
