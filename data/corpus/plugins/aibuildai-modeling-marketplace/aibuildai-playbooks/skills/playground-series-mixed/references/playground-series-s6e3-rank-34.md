# 34th Place Solution — Ridge Multi-View Ensemble

Competition: playground-series-s6e3
Rank: #34
Source: https://www.kaggle.com/c/playground-series-s6e3/writeups/34th-place-solution-ridge-multi-view-ensemble

**Competition:** Playground Series S6E3 — Customer Churn Prediction  
**Task:** Binary classification (ROC-AUC)  
**Dataset:** ~594k rows, synthetic from Telco Customer Churn  
**Final Result:** 34th / 4143+ (Private LB: 0.91831, Public LB: 0.91733, CV: 0.91968)

---
##  Acknowledgements .
-Thanks to @cdeotte, @tilii, @blamerx and @yunsuxiaozi for their contributions. This solution is heavily inspired from their work.

## Introduction

I built many diverse models with different architectures including GBDTs, deep learning, AutoML frameworks, and linear models — each receiving a tailored "feature view" from a shared feature engineering pipeline. All models generated Out-of-Fold (OOF) predictions on a consistent CV strategy, which were then ensembled using **Ridge Regression** as the final meta-learner. The key insight was that Ridge with proper regularization (alpha=1000–2500) was more robust than hill climbing for combining OOFs from heterogeneous sources.

---

## 1. Feature Engineering (~150 features)

Starting from 3 numeric columns (`tenure`, `MonthlyCharges`, `TotalCharges`) and ~16 categorical columns. Many base features were adapted from the public notebook [S6E3|Ridge → XGB + N-gram|0.91927 CV](https://www.kaggle.com/code/blamerx/s6e3-ridge-xgb-n-gram-0-91927-cv) by @blamerx, including derived ratios, frequency encodings, service aggregates, contract features, original-data target probabilities, distribution & quantile features, digit features, n-gram interactions, binning, and the target encoding pipeline. I then added my own features on top of that foundation.

### Features from Public Notebook (credit: @blamerx)
- **Core numerics:** `charges_deviation`, `monthly_to_total_ratio`, `avg_monthly_charges`, frequency encodings, service aggregates, contract features
- **Original-data target probabilities (~20 features):** Churn rate from original dataset merged as `{col}_org_prob`
- **Distribution & quantile features (17 features):** Percentile ranks, z-score gaps, conditional percentile ranks, quantile distances
- **Digit features (35 features):** Digit-level decomposition of numerics (first/last digits, modular arithmetic, fractional parts, deviations). A curated subset of 16 (`DIGIT_GOOD`) was used for models where the full set added noise.
- **N-gram interaction features (16 features):** 15 bi-gram pairs from top 6 categoricals + 1 tri-gram
- **Binning:** `TotalCharges` quantile-binned into 4000 bins via KBinsDiscretizer


### My Features

**Target encoding pipeline:** 10-fold outer CV (StratifiedKFold, seed=42), sklearn `TargetEncoder` with 5-fold inner CV (`smooth='auto'`) fitted on training folds inside cv loop without original data, test TE averaged across all 10 folds

**KMeans Clustering (26 features)**
Fit KMeans with k=25 on the 3 numeric columns (`tenure`, `MonthlyCharges`, `TotalCharges`) using the original dataset. Each row gets a cluster assignment plus Euclidean distances to all 25 centroids, capturing local density patterns in the feature space. The cluster assignment was used as a categorical feature for tree models, while the full distance vectors were fed to neural nets.

**FAISS KNN Features (3 features)**
Built a FAISS GPU index on the original dataset's numerics (StandardScaled), then queried k=20 nearest neighbors for every train/test row:
- `knn_20_churn_rate` — mean churn rate of the 20 nearest original-data neighbors. This is essentially a non-parametric local churn prior — rows surrounded by churners in feature space get a high score.
- `knn_20_min_dist` — distance to the nearest original-data neighbor. Captures how "typical" a synthetic row is relative to the real data distribution.
- `knn_20_mean_dist` — mean distance to 20 neighbors. Rows far from the original data cluster differently than those close to it.

These KNN features were among the most informative for the neural network models, providing a smooth, local signal that complemented the global patterns captured by tree-based features.

---

## 2. Multi-View Feature Strategy

A design choice was giving each model family a **tailored feature subset**. Not all features help all models — CatBoost handles categoricals natively, neural nets prefer normalized inputs, and some models choke on high-cardinality features. The strategy also produced weaker individual models so i would train same models on full available relevant data.

| Model Family | Feature Count | Key Differences |
|---|---|---|
| **XGBoost** | 151 | Full kitchen sink — all digits, all clusters, NUMS_AS_CAT, TE'd categoricals, `enable_categorical=True` |
| **LightGBM** | 104 | Pruned to DIGIT_GOOD (16 best), cluster assignment only (no distances), no NUMS_AS_CAT |
| **CatBoost** | 85 | No target encoding at all — native categorical handling. No ORG_PROBS. CATS + NGRAM passed as `cat_features` |
| **Neural Nets (RealMLP/TabM/ResNet)** | 115 | QuantileTransformer-normalized numerics (`qnorm_*`), full cluster distances, TE'd categoricals |
| **Transformers (FT-Transformer/AutoInt)** | 114 | Only low-cardinality categoricals as embeddings, qnorm numerics |
| **Tree-based DL (NODE/GANDALF/DANet)** | 115 | All categoricals TE'd then dropped, qnorm numerics, full clusters |
| **xRFM** | 80 | Numerics only — no categoricals, no digits, no n-grams |

---

## 3. Models

### Tier 1: GBDTs

**XGBoost** — lr=0.0063, max_depth=5, subsample=0.81, colsample_bytree=0.32, min_child_weight=6, reg_alpha=3.50, reg_lambda=1.29, gamma=0.79, 50k trees with early stopping at 500 rounds. CUDA-accelerated. Also trained Optuna-tuned variants with different hyperparameters.

**LightGBM** — Similar hyperparameters to XGBoost, num_leaves=31. Also trained Optuna-tuned variants.

**CatBoost** — iterations=50k, depth=5, l2_leaf_reg=1.29, random_strength=0.79, Poisson bootstrap, GPU-accelerated.

### Tier 2: Deep Learning (pytabkit)

**RealMLP** — n_ens=8, hidden_sizes=[512, 256, 128], SiLU activation, p_drop=0.05, PLR numerical embeddings (plr_hidden_1=16, plr_hidden_2=8), label smoothing (eps=0.01), robust scaling transforms. Two variants trained.

**TabM** — tabm_k=32, d_block=384, n_blocks=3, dropout=0.1, PLR embeddings. Also a TabM-mini-normal variant (d_block=256, PWL embeddings, dropout=0.2).

**FT-Transformer** — d_token=128, n_layers=3, ffn_dropout=0.2, attention_dropout=0.1, residual_dropout=0.1.

**ResNet-RTDL** — d=256, d_hidden_factor=2, n_layers=4, hidden_dropout=0.2, ReLU + BatchNorm.

**xRFM** — Recursive feature mapping, iters=5, reg=1e-3, max_leaf_samples=30000. Numerics only.

### Tier 2: Deep Learning (pytorch-tabular)

**AutoInt** — Multi-head self-attention for automatic feature interactions. 10-fold CV score: 0.91612.

**NODE** — Neural Oblivious Decision Ensembles. Reduced to 128 trees with input_dropout=0.05 to control overfitting (was 0.98 train vs 0.85 val before tuning). 10-fold CV: 0.91737.

**TabNet** — n_steps=3, sparsemax for row-based feature selection. 10-fold CV: 0.91194.

**GANDALF** — Gated Adaptive Network for Deep Automated Learning of Features. 10-fold CV: 0.91049.


### Tier 1: AutoML Frameworks

**AutoGluon** — `best_quality` and 'extreme_quality' presets with multi-layer stacking, groups='folds'. Trained on the full 152-feature set. Best model (WeightedEnsemble_L2): 0.9194. Generated OOFs from all 16 internal models (CatBoost, LightGBM variants, XGBoost, NeuralNet, RandomForest, ExtraTrees, Transformer and nn variants).

**H2O AutoML** — Ran for 11 hours with `fold_column='fold'` to ensure consistent CV splits with our StratifiedKFold(10, seed=42). Trained 31 models including StackedEnsembles, XGBoost, GBM, GLM, DeepLearning, DRF, XRT. Best (StackedEnsemble_AllModels_4): 0.9171. OOFs extracted from all models via `cross_validation_holdout_predictions()`.

**Note on CV consistency:** H2O and Autogluon were given our explicit fold column, ensuring perfect alignment.

### External Public Models (5 models)
Incorporated OOF predictions from strong public notebooks and trained with my own cv strategy: Neural Network (v312), XGBoost (v102), Logistic Regression with TE (v200), GNN (v1909), and BART-based model (v8400). Thanks to @cdeotte for his valuable contributions.

---

## 4. Ensemble Strategy

### Meta-Learner: Ridge Regression

After experimenting with both Hill Climbing and Ridge Regression, **Ridge proved more robust even though hill climbing was giving me better cv**:

```
Ridge: CV 0.91968, Public LB 0.91733, Private LB 0.91831
Hill Climbing: CV 0.91979, Public LB 0.91723, Private LB 0.91818
```

**Ridge configuration:**
- Alpha: 1000–2500
- Predictions ranked for roc-auc optimization
- StandardScaler applied per fold (zero leakage)
- 10-fold StratifiedKFold (seed=42) on the OOF matrix
- Test predictions averaged across folds

Hill climbing achieved higher CV (0.91979) but lower LB — it overfit the OOF noise through its stochastic search, negative weight support, and power transforms. Ridge's closed-form solution with strong L2 regularization constrained the weights more naturally.

### Model Selection
RFE with Ridge (alpha=500) was used to prune the model pool, selecting the most complementary subset for the final blend.

---

## 5. What Worked

1. **Original dataset as a feature mine** — KNN churn rates, target probabilities, distribution features, and quantile distances from the original data added strong signal without leakage.

2. **Ridge over Hill Climbing** —  Ridge's regularization was more robust for the jungle of models. Hill climbing overfitted despite high CV.

3. **Consistent CV strategy** — Using StratifiedKFold(10, seed=42) across all manually trained models, and explicit fold columns for AutoML frameworks (H2O's `fold_column`).

## 6. What I'd Do Differently

--**Prune weak models earlier** — GANDALF (0.910), xRFM (0.910), TabNet (0.911) added more noise than signal. Should have set a minimum CV threshold (~0.915) for ensemble inclusion. More focus on creating stronger single models since my xgboost was carrying the ensemble, would still like to maintain some weaker models for negative weights.

--**Train models on full dataset** — Multi Features view strategy is good but i would also like to keep model versions trained on all available relevant data w.r.t to its architecture so they can have max signal and hence be individually stronger.

--**No AutoML** — would put more focus on creating real models where i have more control over their cv strategy and overall framework.

--**Milking original data** — i would like to try more features with original data distributions as foundation. 

--**Polynomial Features for ridge** —  Missed out on utilizing polynomial features for oofs while using ridge.

---

This was a fun competition. My robust cv strategy helped me survive the private lb shakeup where i got pushed down 60 places in the public lb on last day of competition. Hopefully we would see a solution to this blind blending menace one day.
