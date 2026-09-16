# 8th Place Solution for S5E10: Predict Road Accident Risk

Competition: playground-series-s5e10
Rank: #8
Source: https://www.kaggle.com/c/playground-series-s5e10/writeups/8th-place-solution-for-s5e10-predict-road-acciden

## Overview 

### 1. Base models

Rather than training hundreds of models and hill-climbing, I focused on a smaller, higher quality set of models and used their OOF predictions as meta-features:

| Model Type             | Variants / Notes                                                                                                                                                                                                                          |
| ---------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **XGBoost**            | Multiple Optuna-tuned versions of @cdeotte’s *“Boosting over Residuals”* ([notebook](https://www.kaggle.com/code/cdeotte/xgb-boosting-over-residuals-cv-0-05595)). Each variant differed in fold count (5–55 folds) and parameter priors. |
| **TabM**               | Three variants of @masayakawamata’s TabM ([discussion](https://www.kaggle.com/competitions/playground-series-s5e10/discussion/610792)), two of which were **converted to predict residuals** over the synthetic data generator.           |
| **HistGBM / LightGBM** | One HistGradientBoostingRegressor and two diverse LightGBM baselines.                                                                                                                                                                     |
| **Neural Networks**    | A small Keras MLP, plus a Neural Network Ensemble (averaging several seeds / architectures).                                                                                                                                              |
| **AutoGluon**          | My own variant of @aliffaagnur’s *AutoGluon Adapter* ([notebook](https://www.kaggle.com/code/aliffaagnur/road-accident-risk-autogluon)).                                                                                                  |

These models were trained using **K-Fold CV (5–55 folds)** with OOF and test predictions saved.

---

### 2. Meta-features and pruning

I stacked all base OOFs (22-48 columns depending on version) into a meta-matrix and applied:

* **Filtering:** Drop models with near-zero variance or OOF RMSE ≫ median (initial clean up).
* **Duplicate removal:** Eliminate columns with |ρ| > 0.9995, keeping the stronger one.
After this first stage pruning, the final subset of models used for stacking was chosen via Greedy NNLS and LassoCV selection, which identify the most complementary OOF features.   

This tended to reduce the feature set from ~40 → 4–6 columns.

---

### 3. Meta-learning and ensembling

| Stage | Model / Technique                   | Description                                                                                                         |
| ----- | ----------------------------------- | ------------------------------------------------------------------------------------------------------------------- |
| 1  | **RidgeCV Sweep**                   | Tuned α ∈ {0.01 … 10}; produced top single-model CV (0.05579).                                                      |
| 2  | **MetaNN**                          | Small BN → Dense(32,GELU) → Dropout → Dense(1), AdamW + CosineRestarts, Huber loss; 7-fold stratified CV × 3 seeds. |
| 3   | **Residual MetaNN**                 | Trained on *(y − ridge OOF)* and added back to ridge to capture non-linear leftovers.                               |
| 4   | **Greedy NNLS & LassoCV selection** | Identified minimal model subsets and positive blending weights.                                                     |
| 5   | **Final Blends**                    | • NNLS of {MetaNN, Ridge} • 2-stage Ridge on [ridge, MetaNN].                                                       |


---

### 4. Evaluation and results

| Candidate Submission             | CV (OOF RMSE)   | LB Score                                |
| -------------------------------- | ----------------- | --------------------------------------- |
| Ridge                            | 0.05579           | **0.05564 (8th Place)**                |
| Two-stage Ridge (Ridge + MetaNN) | 0.05579           | 0.05563 (would have been even better !) |
| NNLS Blend                       | 0.05579           | ≈ 0.05564                               |
| Greedy / Lasso Subsets           | 0.05579 – 0.05580 | ≈ 0.05565                               |


---

## Code Highlights

**Prune Weak OOF Features Before Stacking**

```python
# Per-column diagnostics
def rmse(a,b):
    return mean_squared_error(a, b, squared=False)

stats = []
for c in X_meta.columns:
    p = X_meta[c].values
    stats.append((c, rmse(y, p), np.std(p), np.corrcoef(p, y)[0,1]))

stats_df = (pd.DataFrame(stats, columns=["col","oof_rmse","std","corr_y"]).sort_values("oof_rmse"))

# Filter 
bad_cut = stats_df.oof_rmse.median() + 0.02
bad_cols = set(stats_df.loc[(stats_df["std"]<1e-6) | (stats_df["oof_rmse"]>bad_cut), "col"])

keep = [c for c in X_meta.columns if c not in bad_cols]

# Drop near dupes by corr
C = X_meta[keep].corr().values
drop_dupes, thr = set(), 0.9995
for i in range(len(keep)):
    if keep[i] in drop_dupes: 
        continue
    for j in range(i+1, len(keep)):
        if keep[j] in drop_dupes: 
            continue
        if abs(C[i,j]) > thr:
            rm_i = stats_df.loc[stats_df.col==keep[i], "oof_rmse"].iloc[0]
            rm_j = stats_df.loc[stats_df.col==keep[j], "oof_rmse"].iloc[0]
            drop_dupes.add(keep[j] if rm_j >= rm_i else keep[i])

sel_cols0 = [c for c in keep if c not in drop_dupes]
X_USED, X_USED_TEST = X_meta[sel_cols0].copy(), X_meta_test[sel_cols0].copy()

```

**Tune Ridge Meta-Learner Over Pruned Feature Set**

```python
def cv_ridge_preds(X, y, X_test, alpha, n_splits=7, seed=42, collect_coefs=False):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof = np.zeros(len(y)); test_pred = np.zeros(len(X_test))
    coefs = []
    for tr, va in kf.split(X):
        m = Ridge(alpha=alpha, fit_intercept=True, random_state=seed)
        m.fit(X.iloc[tr], y.iloc[tr])
        oof[va] = m.predict(X.iloc[va])
        test_pred += m.predict(X_test) / n_splits
        if collect_coefs: coefs.append(m.coef_)
    return rmse(y, oof), oof, test_pred, (np.vstack(coefs) if collect_coefs else None)

best_rmse = 1e9
for a in [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0]:
    r, oof_r, pred_r, coef_arr = cv_ridge_preds(X_USED, y, X_USED_TEST, a, collect_coefs=True)
    if r < best_rmse:
        best_rmse, best_alpha = r, a
        best_oof_ridge, best_pred_ridge, best_coef_arr = oof_r, pred_r, coef_arr

```

---

## Sources and Acks

* [@cdeotte – XGB Boosting over Residuals CV 0.05595](https://www.kaggle.com/code/cdeotte/xgb-boosting-over-residuals-cv-0-05595)
* [@siukeitin (broccoli beef) – Bayesian Optimal Solution](https://www.kaggle.com/siukeitin)
* [@masayakawamata – TabM Baseline and Discussion](https://www.kaggle.com/competitions/playground-series-s5e10/discussion/610792)
* [@aliffaagnur – AutoGluon Adapter Notebook](https://www.kaggle.com/code/aliffaagnur/road-accident-risk-autogluon)

---

Thanks to everyone for a great playground competition!
