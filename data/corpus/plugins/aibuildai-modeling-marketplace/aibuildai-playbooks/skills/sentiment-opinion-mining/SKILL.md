---
description: >-
  ML playbook for sentiment opinion mining competitions. Use when tackling a Kaggle-style competition involving sentiment opinion mining. Teaches how to reason about trust your cross-validation over public leaderboard, match text preprocessing to model architecture, prevent data leakage between training and validation, leverage external dataset diversity for robustness. 16 top-solution writeups across 2 competitions: crowdflower-weather-twitter (rank #1) and jigsaw-toxic-severity-rating (ranks #1-#7, #9-#12, #14, #24, #27, #41)
---

# Sentiment and Opinion Mining Playbook

Sentiment and opinion mining tasks involve extracting subjective information from text — whether multi-label confidence prediction (sentiment + attributes) or severity ranking (toxicity ordering). The core challenge is that these tasks sit at the intersection of linguistic nuance and human disagreement: what one annotator considers toxic, another may not, and models must learn robust patterns that generalize across diverse annotation perspectives and text domains.

**Source material:** 16 top-solution writeups across 2 competitions: crowdflower-weather-twitter (rank #1) and jigsaw-toxic-severity-rating (ranks #1-#7, #9-#12, #14, #24, #27, #41)

## Principle Index

| # | Principle | Consensus | File |
|---|-----------|-----------|------|
| 1 | Trust Your Cross-Validation Over Public Leaderboard | 14/15 Jigsaw solutions explicitly mentioned this principle | principles/01.md |
| 2 | Match Text Preprocessing to Model Architecture | 3/3 solutions that compared cleaning strategies, with 2nd place showing 0.81363 without cleaning vs 0.80826 with cleaning | principles/02.md |
| 3 | Prevent Data Leakage Between Training and Validation | 7/7 top solutions that used validation_data.csv removed overlapping texts from training | principles/03.md |
| 4 | Leverage External Dataset Diversity for Robustness | 15/15 Jigsaw solutions used 2+ external datasets; top-3 used 3-5 datasets each | principles/04.md |
| 5 | Apply Multi-Stage Fine-Tuning from General to Specific | 4/15 top solutions explicitly used multi-stage fine-tuning, with 14th place detailing a 5-stage pipeline | principles/05.md |
| 6 | Choose Loss Function Based on Label Type and Task Goal | 12/15 solutions discussed loss; 8 used MarginRankingLoss for pairwise tasks, 4 used MSE for regression | principles/06.md |
| 7 | Optimize Ensemble Weights on Validation Data with Diversity Awareness | 10/15 solutions used automated weight optimization (Optuna, scipy.optimize, genetic algorithms) | principles/07.md |
| 8 | Use Pre-Trained Toxicity Models as Feature Extractors | 5/15 solutions used Detoxify or similar pre-trained models, with 3rd place using 11 Detoxify features + 1 TF-IDF | principles/08.md |
| 9 | Train Linear Models with TF-IDF for Fast, Interpretable Baselines | 8/15 solutions included TF-IDF+Ridge models, with multiple teams reporting 0.68-0.70 CV as baselines | principles/09.md |
| 10 | Implement High-Frequency Validation During Training | 3/15 solutions mentioned validating every 10% of an epoch for large datasets | principles/10.md |
| 11 | Apply Layerwise Learning Rate Decay for Transformer Fine-Tuning | 5/15 solutions mentioned layerwise LR, with typical ranges 2e-5 (backbone) to 1e-3 (head) | principles/11.md |
| 12 | Use Ranking Transformation for Ensemble Compatibility | 4/15 solutions explicitly used rank transformation (rankdata) before ensembling | principles/12.md |
| 13 | Balance Dataset Diversity and Model Diversity in Ensembles | 4th place explicitly weighted dataset diversity 1/3, model diversity 2/3; multiple solutions trained same architecture on different datasets | principles/13.md |
| 14 | Monitor and Mitigate Overfitting with Multiple Techniques | The 2nd-place jigsaw-toxic-severity-rating solution warned that the public leaderboard could mislead teams into overfitting, and relied on weight decay, dropout, and early stopping | principles/14.md |
| 15 | Create Pairwise Training Data from Multi-Label Annotations | 5/5 solutions that used Ruddit or Jigsaw data for ranking tasks created pairwise comparisons | principles/15.md |

## How to Use

Use Glob to list principles/*.md, then Read the principles relevant to your task. The references/ directory holds the raw ranked writeups these principles were distilled from.

These principles work together as a system: start by establishing leak-free validation and trusting it over public LB, diversify your data and models to capture complementary signals, match your preprocessing and loss to your model type and task structure, and ensemble carefully with normalization and weight optimization. The path from baseline to winning solution is incremental refinement of each component, not a single silver bullet.
