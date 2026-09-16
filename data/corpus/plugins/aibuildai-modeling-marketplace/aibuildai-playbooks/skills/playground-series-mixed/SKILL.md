---
description: >-
  ML playbook for playground series mixed competitions. Use when tackling a Kaggle-style competition involving playground series mixed. Teaches how to reason about start simple and add complexity only when justified by cv, trust cross-validation over public leaderboard, match categorical encoding to model architecture, leverage original datasets for synthetic competitions. 77 playground series competitions with 348 top-solution writeups spanning categorical encoding challenges, small-sample overfitting tasks, time series forecasting, and multi-class classification.
---

# Tabular Playground Series Playbook

Tabular Playground Series competitions feature diverse tabular machine learning tasks including binary/multi-class classification, regression, and time series forecasting on synthetically-generated or simplified datasets. The core challenge is balancing model complexity with generalization—these datasets often reward simple, well-regularized approaches over complex architectures, and cross-validation reliability is paramount since public/private leaderboard splits can differ significantly.

**Source material:** 77 playground series competitions with 348 top-solution writeups spanning categorical encoding challenges, small-sample overfitting tasks, time series forecasting, and multi-class classification.

## Principle Index

| # | Principle | Consensus | File |
|---|-----------|-----------|------|
| 1 | Start Simple and Add Complexity Only When Justified by CV | Mentioned in 15+ top solutions across different competition types. Simple models (logistic regression, linear models) won multiple competitions outright. | principles/01.md |
| 2 | Trust Cross-Validation Over Public Leaderboard | Emphasized in 20+ writeups, including multiple 1st place solutions. The TPS Jan-2022 winner had public LB rank 306 but won on private LB by trusting CV. | principles/02.md |
| 3 | Match Categorical Encoding to Model Architecture | 5/5 top cat-in-the-dat solutions used this principle. Tree-based models used ordinal encoding, linear models used one-hot or target encoding. | principles/03.md |
| 4 | Leverage Original Datasets for Synthetic Competitions | Mentioned in 8+ TPS writeups. The playground-series-s6e5 winner explicitly used original data with sample weights. | principles/04.md |
| 5 | Exploit Data Generation Artifacts in Synthetic Datasets | 2 competition-winning solutions (TPS Feb-2022 1st place, playground s3e1 2nd place) explicitly exploited generation flaws. | principles/05.md |
| 6 | Invest in Domain-Specific Feature Engineering Over Model Tuning | 10+ time series solutions emphasized this. The TPS Jan-2022 winner spent most effort on holiday/seasonal features, not hyperparameters. | principles/06.md |
| 7 | Use Stratified CV for Imbalanced or Multiclass Tasks | Mentioned in 12+ solutions across classification tasks. Standard practice for all top solutions in imbalanced competitions. | principles/07.md |
| 8 | Apply Heavy Regularization for Small or Sparse Datasets | The dont-overfit-ii winners all used regularization. Multiple small-sample TPS solutions emphasized L1/L2 penalties. | principles/08.md |
| 9 | Optimize Hyperparameters with Optuna for Gradient Boosting | Mentioned in 10+ solutions. Optuna was the most common hyperparameter optimization tool across TPS competitions. | principles/09.md |
| 10 | Build Ensemble Diversity Through Preprocessing and Model Families | The Otto 1st place used 33 diverse base models; TPS Sep-2021 2nd place used 115 models with different preprocessing. | principles/10.md |
| 11 | Use Simple Meta-Models for Stacking | 15+ solutions used Ridge, Lasso, or Logistic regression as meta-models. Complex meta-models were rare and often underperformed. | principles/11.md |
| 12 | Use Rank Transformation When Blending for AUC/Rank Metrics | 2 top solutions explicitly recommended rank blending for AUC tasks. | principles/12.md |
| 13 | Clip Probabilities to Avoid Log-Loss Extremes | 3 top log-loss solutions explicitly clipped predictions. Standard practice for log-loss competitions. | principles/13.md |
| 14 | Consider AutoML for Fast Competitive Baselines | 4+ solutions used AutoGluon successfully, including a 1st place win. Multiple solutions used H2O, LightAutoML. | principles/14.md |
| 15 | Apply Pseudo-Labeling Cautiously with Confidence Thresholding | 3 solutions used pseudo-labeling successfully, but emphasized careful thresholding and validation. | principles/15.md |

## How to Use

Use Glob to list principles/*.md, then Read the principles relevant to your task. The references/ directory holds the raw ranked writeups these principles were distilled from.

These principles work together in stages: start simple with proper CV and categorical encoding, invest in feature engineering over model complexity, apply regularization for small datasets, tune hyperparameters efficiently, and build diverse ensembles with simple meta-models. Trust your CV throughout, and use public LB only as a sanity check.
