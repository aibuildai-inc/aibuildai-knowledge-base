# # 25th Place Solution

Competition: playground-series-s6e2
Rank: #25
Source: https://www.kaggle.com/c/playground-series-s6e2/writeups/25th-place-solution

## Overview

Final ensemble OOF ROC-AUC: **0.955805** | PB: **~0.95533**

This was a fun competition. Here's what worked.

---

## Feature Engineering

The core idea was to enrich each sample with **group statistics derived from the original Heart Disease dataset**, specifically the mean, median, std, skew, and count of the target per feature group (e.g., `orig_Age_mean`, `orig_Thallium_std`, etc.).

Additional features used:

- **SIN/COS periodic encodings** for `BP`, `Cholesterol`, `Max HR`, and `ST depression`
- **COMBO_ST_Slope**, interaction between `ST depression` and `Slope of ST`
- **Chest_asymptomatic**, binary flag for `Chest pain type == 4`
- **Frequency encoding** for numerical columns (computed from training fold only)
- **Target encoding** via sklearn's `TargetEncoder` with `cv=10` inner folds
- **CAT_** prefix columns, stringified numerical columns passed to models as categoricals

Not every model uses all of these features, each model family has a different appetite for feature types, as explained below.

---

## Models

We trained a diverse set of base models, each with 10-fold stratified CV.

**Pseudo-labeling** (teacher-student): the teacher model predicts on the test set; samples with confidence > 0.999 or < 0.001 are added to student training with hard labels. This was used for XGBoost and CatBoost variants.

### XGBoost
Features: base features + group stats + SIN/COS + COMBO_ST_Slope + Chest_asymptomatic + frequency encoding + target encoding + CAT_ columns. Numerical columns are passed as-is; CAT_ columns are cast to XGBoost's native `category` dtype via `enable_categorical=True`. Multiple seeds trained, with and without teacher-student pseudo-labeling at threshold 0.999.

### CatBoost
Same feature set as XGBoost. CatBoost handles categoricals natively so CAT_ columns are passed as categorical indices directly. Multiple seeds, with and without pseudo-labeling.

### LightGBM
Same feature set as XGBoost and CatBoost. Standard CV, no pseudo-labeling.

### RealMLP (pytabkit)
Features: base features + group stats + SIN/COS + COMBO_ST_Slope + Chest_asymptomatic + frequency encoding + target encoding + CAT_ columns. Categorical columns are passed as strings via `cat_col_names`; RealMLP handles its own embedding internally. This was the **best solo model family** across all seeds.

### TabCA-PLE (custom)
Features split into two streams, **categorical** (base categoricals + CAT_ columns) and **numerical** (all numeric including group stats, SIN/COS, frequency, target encoding). Numerical features are encoded with **Piecewise Linear Encoding (PLE)** using decision tree-derived bin boundaries, then the two streams interact via **cross-attention**: categorical tokens attend to numerical embeddings before being concatenated and passed through a residual MLP.

### MLP-PLR
Features: same split as TabCA-PLE. Numerical features use periodic + linear + ReLU embeddings (PLR). Simpler architecture than TabCA-PLE with no cross-attention.

### MLP-RTDL
Features: same as MLP-PLR. Standard RTDL-style MLP with periodic embeddings. No special cross-feature interaction.

### TabM
Features: same as other neural nets. Parallel PLE periodic variant, runs multiple sub-networks over different feature subsets in parallel.

### ResNet
Features: base features + group stats + frequency encoding + target encoding. Standard tabular ResNet, no SIN/COS or special embeddings.

### TabNet
Features: base features + group stats + frequency encoding + target encoding. Standard TabNet with attention-based feature selection.

### Ridge Ensemble
Rather than training on raw features, this model uses the **OOF predictions of other models as input features** and fits a Ridge regression on top. Two variants: one using all base model OOFs, one using target-stacked OOFs. Useful as a meta-learner to correct systematic errors across base models.

### AutoGluon
AutoGluon was included in the pool but intentionally **not tuned or optimized**. We ran it with minimal configuration as a quick baseline to see if its internal ensembling could add diversity. It couldn't, the CV of 0.93333 was far below all other models, and spending time optimizing AutoGluon would have taken away time from training more targeted models. At this competition's level, a poorly configured AutoGluon adds noise rather than signal.

---

## Ensemble

For blending, we used **hill climbing** on OOF predictions.

We ran hill climbing in two modes, **positive weights only** and **allowing negative weights**. With positive weights only the ensemble OOF reached **0.955797**. Allowing negative weights pushed it to **0.955805**. However, the private LB score was identical for both, meaning the marginal CV gain from negative weights didn't translate to any real difference on unseen data.

Individual model OOF scores (10-fold CV):

| Model | OOF ROC-AUC |
|---|---|
| RealMLP seed 10 (pseudo) | 0.95577 ⭐ best solo |
| RealMLP seed 60 (pseudo) | 0.95576 |
| RealMLP seed 4 | 0.95576 |
| RealMLP seed 13 | 0.95576 |
| Ridge Ensemble | 0.95575 |
| RealMLP seed 7 | 0.95575 |
| RealMLP seed 2 | 0.95574 |
| Ridge Ensemble (target-stacked) | 0.95573 |
| CatBoost seed 24 | 0.95572 |
| CatBoost seed 6 | 0.95572 |
| CatBoost seed 25 | 0.95568 |
| CatBoost seed 12 | 0.95566 |
| XGBoost seed 16 (pseudo) | 0.95566 |
| CatBoost seed 17 (pseudo) | 0.95565 |
| TabM seed 22 | 0.95564 |
| CatBoost seed 7 | 0.95561 |
| CatBoost seed 10 | 0.95561 |
| TabNet seed 8 | 0.95558 |
| XGBoost seed 40 | 0.95557 |
| XGBoost seed 11 | 0.95557 |
| XGBoost seed 4 | 0.95553 |
| XGBoost seed 5 | 0.95551 |
| CatBoost seed 3 | 0.95550 |
| BARTZ Balanced seed 18 | 0.95544 |
| ResNet seed 2 | 0.95541 |
| LightGBM seed 13 | 0.95541 |
| XGBoost DAE seed 41 | 0.95529 |
| XGBoost TabM seed 11 (pseudo) | 0.95504 |
| MLP-PLR seed 8 | 0.95500 |
| MLP-RTDL seed 23 | 0.95482 |
| TabCA-PLE seed 20 | 0.95478 |
| AutoGluon seed 19 | 0.93333 |

Final ensemble OOF: **0.955805**

---

## Models Not Selected by Hill Climbing

Some models were in the pool but failed to improve the ensemble regardless of weight sign:

- **AutoGluon seed 19**, CV of 0.93333 was far below the rest and its error pattern was too noisy to be useful even as a corrector
- **TabCA-PLE seed 20** and **MLP-RTDL seed 23**, lowest CV among neural networks (~0.9548) and predictions too correlated with stronger RealMLP variants to add meaningful diversity
- **MLP-PLR seed 8**, largely redundant with the RealMLP family despite a different architecture on paper
- **XGBoost TabM seed 11 (pseudo)**, didn't generalize as well as the standard XGBoost pseudo variants

The common pattern: models that were both **weak in CV** and **not diverse enough** relative to stronger models already in the ensemble didn't make the cut.

---

## Key Takeaway

**Trust your CV.** Hill climbing with negative weights gave a marginal CV gain (0.955797 → 0.955805) but the private LB was identical, a good reminder that very small CV differences at this level don't always mean anything real.

Thanks to everyone in the community, it is impossible to mention you all one by one, but every discussion, notebook, and comment made a difference. Special thanks to @cdeotte, @davidholzmueller, @masayakawamata, @tilii7, @mahoganybuttstrings, @mirko45, @ravi20076, @omidbaghchehsaraei, and @optimistix for keeping the discussion alive and making this competition so much more enjoyable.
