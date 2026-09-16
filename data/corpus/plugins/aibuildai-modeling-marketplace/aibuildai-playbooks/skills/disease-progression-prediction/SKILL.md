---
description: >-
  ML playbook for disease progression prediction competitions. Use when tackling a Kaggle-style competition involving disease progression prediction. Teaches how to reason about assess feature-to-sample ratio before feature engineering, mine temporal metadata before complex features, identify hidden subpopulations through eda, build robust cv for small, noisy data. 57 top-solution writeups across 5 competitions (AMP Parkinson's, Child Mind Institute PIU, ICR Age-Related Conditions, CIBMTR post-HCT Survival, Practice Fusion Diabetes)
---

# Disease Progression Prediction Playbook

Disease progression prediction challenges you to forecast clinical outcomes over time using limited patient data. The distinctive feature is the intersection of small sample sizes, high-dimensional medical measurements, temporal dynamics, and significant noise. The core challenge is extracting reliable signal when the curse of dimensionality threatens to turn every feature into noise.

**Source material:** 57 top-solution writeups across 5 competitions (AMP Parkinson's, Child Mind Institute PIU, ICR Age-Related Conditions, CIBMTR post-HCT Survival, Practice Fusion Diabetes)

## Principle Index

| # | Principle | Consensus | File |
|---|-----------|-----------|------|
| 1 | Assess Feature-to-Sample Ratio Before Feature Engineering | 5/5 top Parkinson's solutions, 3/3 top ICR solutions explicitly discussed this | principles/01.md |
| 2 | Mine Temporal Metadata Before Complex Features | Top 18/18 Parkinson's gold solutions used visit date features; 1st place ignored all biomarker data | principles/02.md |
| 3 | Identify Hidden Subpopulations Through EDA | 5th place Parkinson's solution found control groups or patient strata | principles/03.md |
| 4 | Build Robust CV for Small, Noisy Data | All top-3 CMI solutions used 100+ repeated folds; all Parkinson's/ICR solutions used group/stratified CV | principles/04.md |
| 5 | Trust CV Over Public Leaderboard | 3rd CMI: 'focused entirely on CV after LB correlation broke' | principles/05.md |
| 6 | Use Simple Models with Strong Regularization | 5th Parkinson's used 2-feature linear/isotonic; 1st equity-HCT found optimal depth=2 for XGB/LGB | principles/06.md |
| 7 | Apply Data Augmentation for Tabular Small Data | 3/3 top CMI solutions, multiple ICR solutions used augmentation | principles/07.md |
| 8 | Handle Missing Data with Domain-Driven Imputation | 2nd/4th CMI solutions explicitly modeled missingness; multiple solutions treated NaN as informative | principles/08.md |
| 9 | Optimize Post-Processing and Thresholds Separately | All CMI top solutions optimized sii thresholds; 1st Parkinson's picked class minimizing SMAPE per prediction | principles/09.md |
| 10 | Engineer Target Transformations When Direct Prediction Fails | 1st CMI predicted PCIAT_Total→sii; 1st equity-HCT normalized rank within efs groups; 1st Parkinson's used classification instead of regression | principles/10.md |
| 11 | Separate Model Types by Task Structure | 1st equity-HCT trained separate classifier/regressor | principles/11.md |
| 12 | Quantile Binning to Handle Noisy Continuous Features | 1st CMI applied quantile binning to 'a good chunk of features'; multiple solutions discretized features | principles/12.md |
| 13 | Build Interpretable Features from Actigraphy Time-Series | All CMI top solutions extracted summary statistics; 4th place found 'longest inactivity streaks' | principles/13.md |
| 14 | Validate Train-Test Distribution Shifts | Parkinson's solutions noted test set differences | principles/14.md |

## How to Use

Use Glob to list principles/*.md, then Read the principles relevant to your task. The references/ directory holds the raw ranked writeups these principles were distilled from.

These principles work together: start with EDA to find subpopulations and temporal patterns, build robust CV, keep models simple, and validate against distribution shifts. The winning pattern is often the simplest model on the cleanest signal, not the most sophisticated architecture on all available features.
