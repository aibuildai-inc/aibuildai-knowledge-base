---
description: >-
  ML playbook for rating ranking prediction competitions. Use when tackling a Kaggle-style competition involving rating ranking prediction. Teaches how to reason about optimize directly for the evaluation metric, identify and respect data structure for cross-validation, engineer heavy aggregation features for sequential tabular data, handle class imbalance with weighted metrics and sampling. Analysis of 165 top-solution writeups across 23 competitions including AmEx Default Prediction, WSDM Chatbot Arena, March Madness, Mercari Price Suggestion, Porto Seguro, and others.
---

# Rating and Ranking Prediction Playbook

Rating and ranking prediction encompasses predicting continuous scores, probabilities, or relative orderings across diverse domains: credit default risk, product prices, game-playing strength, tournament outcomes, chatbot preferences, and insurance claims. The core challenge is accurately modeling the relationship between features and a target rating/ranking while handling class imbalance, temporal drift, and domain-specific constraints. Success requires metric-aware optimization, heavy feature engineering for tabular data, and carefully constructed ensembles.

**Source material:** Analysis of 165 top-solution writeups across 23 competitions including AmEx Default Prediction, WSDM Chatbot Arena, March Madness, Mercari Price Suggestion, Porto Seguro, and others.

## Principle Index

| # | Principle | Consensus | File |
|---|-----------|-----------|------|
| 1 | Optimize Directly for the Evaluation Metric | Nearly all top solutions explicitly optimize for the competition metric rather than a proxy | principles/01.md |
| 2 | Identify and Respect Data Structure for Cross-Validation | Top solutions in temporal competitions use time-aware CV; most in grouped data use group-aware splits | principles/02.md |
| 3 | Engineer Heavy Aggregation Features for Sequential Tabular Data | Top tabular solutions (AmEx 1st/2nd/3rd, Elo 7th, Porto Seguro 9th) use 3k-7k features from aggressive aggregation | principles/03.md |
| 4 | Handle Class Imbalance with Weighted Metrics and Sampling | All top solutions in imbalanced tasks (AmEx 20:1 downsampling, Porto Seguro 3.6% positive rate) use sample weights or resampling | principles/04.md |
| 5 | Normalize Features by Temporal Subsets to Handle Data Drift | AmEx 2nd place, Elo 7th, and temporal competitions explicitly normalize by month/season to handle drift | principles/05.md |
| 6 | Diversify Ensemble Components Across Features, Preprocessing, and Architectures | Top solutions across competitions use ensembles; most explicitly create diversity via different feature sets or preprocessing | principles/06.md |
| 7 | Use Deep Sequence Models for Long Temporal Dependencies | Several top solutions in sequence-heavy tasks (student performance, transaction histories) use RNN/LSTM/Transformer over flattened features | principles/07.md |
| 8 | Apply Rigorous Feature Selection to Reduce Overfitting | AmEx 2nd reduced 7k -> 2.5k features via permutation importance; Porto Seguro top solutions filter zero-importance features | principles/08.md |
| 9 | Distill Large Models into Small Ones for Inference Constraints | WSDM chatbot arena 1st (five Qwen2.5-72B fold teachers distilled into a Qwen2.5-14B student with soft labels) and 3rd (distillation raised a 14B model's CV from 0.705 to 0.717) | principles/09.md |
| 10 | Integrate Domain-Specific External Features and Rankings | March Madness top solutions use ELO ratings, Nate Silver rankings; tournament solutions universally use external rankings | principles/10.md |
| 11 | Blend Ensembles with Optimized Weights, Not Simple Averaging | Most top ensemble solutions use weighted blending (optimized via Lasso, Optuna, or hill climbing) over simple averaging | principles/11.md |
| 12 | Calibrate Probabilities for Probability-Sensitive Metrics | Some top solutions in logloss competitions (Prudential 1st, Porto Seguro 9th) apply post-hoc calibration | principles/12.md |
| 13 | Use Test-Time Augmentation (TTA) for Robustness | WSDM chatbot arena 1st/2nd/3rd/12th use swap-order TTA, often only on the most uncertain or shortest samples to fit the time limit | principles/13.md |
| 14 | Treat Regression as Classification with Soft Targets for Robustness | Mercari 1st place used classification with soft bucketed targets; Prudential 1st place used ordinal regression | principles/14.md |

## How to Use

Use Glob to list principles/*.md, then Read the principles relevant to your task. The references/ directory holds the raw ranked writeups these principles were distilled from.

These principles are complementary, not exclusive. A strong solution often combines heavy aggregation features with sequence models (principle 3 + 7), diverse ensembles (6) optimized with weighted blending (11), calibrated outputs (12), and TTA (13). Start by understanding your metric (1) and data structure (2), then layer in techniques that match your task characteristics and constraints.
