# 20th Place Solution: NeurIPS - Open Polymer Prediction 2025

Competition: neurips-open-polymer-prediction-2025
Rank: #20
Source: https://www.kaggle.com/c/neurips-open-polymer-prediction-2025/writeups/20th-place-solution-neurips-open-polymer-predictio

## Introduction

Thank you for organizing this competition. I wrote the code while revisiting my early machine-learning mindset. Because the Public LB was based on 8% of the test set, I expected a shake-up and focused on strengthening CV. That helped me move from 407th to 20th.

Final submission: CV = 0.0436, Public LB = 0.065, Private LB = 0.085

## Validation

- Tried several schemes and adopted Stratified K-Fold CV (k=5), which correlated best with the Public LB.
- Trained a separate model for each of the five targets, each with its own validation.
  - Stratification by binning each target into five equal-frequency bins (20% each).

## Feature Engineering

- Added every feature inspired by “POINT2: A Polymer Informatics Training and Testing Database” that improved CV and LB:
  - MACCS Keys
  - Morgan fingerprints
  - RDKit fingerprints
  - Atom pair fingerprints
  - Topological torsion fingerprints
  - Graph-based descriptors
  - Polymer structural features
  - AUTOCORR2D
- Feature filtering:
  - Constant features
  - Highly correlated features

Final feature set: 1,072 columns.

## Modeling

- Incrementally added models that improved CV and LB. Weights were determined per target automatically.
  - Tg: xgb=0.789, knn=0.112, cat=0.052, hist=0.039, et=0.008, lgb=0.000
  - FFV: xgb=0.385, hist=0.273, lgb=0.218, knn=0.123, cat=0.000, et=0.000
  - Tc: lgb=0.643, et=0.167, hist=0.108, cat=0.082, xgb=0.000, knn=0.000
  - Density: cat=0.327, xgb=0.256, et=0.213, lgb=0.147, hist=0.058, knn=0.000
  - Rg: cat=0.401, et=0.250, xgb=0.196, lgb=0.094, hist=0.058, knn=0.000

## Post-processing

- For each target, aligned predictions to the training distribution by matching the mean or the standard deviation. On the Public LB these helped:
  - Tg: mean matching
  - FFV: mean and standard-deviation matching
    
Other adjustments degraded performance, so they were not used.

## References

- [“POINT2: A Polymer Informatics Training and Testing Database”](https://arxiv.org/abs/2503.23491)
- [“The Science of Shake-ups”(Japanese)](https://speakerdeck.com/rsakata/shake-upwoke-xue-suru) by @rsakata 
- [“Materials Informatics: Practical Handbook”(Japanese)](https://www.amazon.co.jp/%E3%83%9E%E3%83%86%E3%83%AA%E3%82%A2%E3%83%AB%E3%82%BA%E3%83%BB%E3%82%A4%E3%83%B3%E3%83%95%E3%82%A9%E3%83%9E%E3%83%86%E3%82%A3%E3%82%AF%E3%82%B9-%E5%AE%9F%E8%B7%B5%E3%83%8F%E3%83%B3%E3%83%89%E3%83%96%E3%83%83%E3%82%AF-%E9%AB%98%E5%8E%9F%E6%B8%89/dp/4627858418) by @mipypf
