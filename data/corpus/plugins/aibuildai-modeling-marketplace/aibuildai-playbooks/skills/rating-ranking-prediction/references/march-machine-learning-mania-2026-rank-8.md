# 8th PLACE - MARCH MACHINE LEARNING MANIA 2026

Competition: march-machine-learning-mania-2026
Rank: #8
Source: https://www.kaggle.com/c/march-machine-learning-mania-2026/writeups/8th-place-march-machine-learning-mania-2026

# TOP 8 NCAA 2026 March Machine Learning Mania — Solution Write-Up

---

## 1. Executive Summary

### Team Composition

### High-level Approach Overview

This solution predicts the win probability for every possible matchup in the 2026 NCAA Men's and Women's Basketball Tournaments (519,144 total matchup pairs). The pipeline consists of four major stages:

1. **Data Cleaning & Temporal Filtering** — A "Time-Travel Filter" removes all future data to prevent leakage, simulating the exact information available before Selection Sunday.
2. **Advanced Feature Engineering** — 22 features per team (20 Sabermetrics + Elo + GLM Quality) are computed from Regular Season box scores, then converted into pairwise Delta features.
3. **Hybrid Ensemble Modeling** — XGBoost and LightGBM regressors are trained with season-grouped cross-validation and time-decay sample weighting.
4. **Calibrated Inference** — Final probabilities are clipped to `[0.02, 0.98]` to guard against extreme Brier Score penalties on upset outcomes.

### Key Differentiators

- **Time-Travel Filter** that strictly enforces temporal boundaries (DayNum ≤ 118 for 2026), completely removing future Seeds/Slots data.
- **Dual-Rating System**: Elo (long-term historical strength) + GLM Quality (current-season intrinsic strength).
- **20 derived Sabermetrics** computed from raw box-score columns — none exist in the original dataset.
- **Symmetric Data Augmentation**: each historical tournament game generates two training rows with negated deltas, enforcing `P(A>B) + P(B>A) ≈ 1`.
- **RobustScaler** (median/IQR-based) instead of StandardScaler — preserves legitimate outliers that are characteristic of March Madness upsets.
- **GroupKFold by Season** cross-validation ensures the model is evaluated on its ability to predict truly unseen future seasons.

---

## 2. Methodology

### Problem Formulation

The problem is formulated as a **regression** task predicting continuous probabilities in `[0, 1]`, optimized with `reg:squarederror` (XGBoost) and `regression` (LightGBM) objectives. This is preferred over binary classification because the evaluation metric (Brier Score) directly measures the quality of probability estimates, not just binary accuracy.

### Model Selection Philosophy

A **Hybrid Ensemble** of XGBoost and LightGBM was chosen for several reasons:
- Both are state-of-the-art gradient boosting frameworks capable of capturing non-linear interactions between features.
- Ensembling two different GBDT implementations reduces variance — they use different tree-building heuristics (exact greedy vs. histogram-based), default regularization, and leaf-growth strategies.
- Simple averaging of predictions is used (no meta-learner/stacking) to minimize overfitting risk on a relatively small training set (~2,900 men's + ~1,900 women's matchup rows after augmentation).

### Target Variable Definition

- `Target = 1` if TeamA (the team listed first in the matchup ID) wins.
- `Target = 0` otherwise.
- Due to symmetric augmentation, every historical game produces both a `Target=1` and `Target=0` row with inverted Delta features, ensuring balanced classes.

### Handling Seasonality and Tournament Variance

- **Time-Decay Weighting**: `W = exp(-0.15 × (2026 − Season))` assigns exponentially higher importance to recent seasons during training.
- **GroupKFold by Season**: Cross-validation folds are split by entire seasons, never mixing games from the same season across train/validation, preventing temporal leakage.
- **Outlier Preservation**: March Madness is defined by upsets. Outlier game statistics are intentionally retained (not clipped or removed) and handled via RobustScaler.

---

## 3. Data Sources & Preprocessing

### Official Kaggle Dataset Usage

All data comes from the official Kaggle competition dataset "March Machine Learning Mania 2026":

| File | Purpose |
|------|---------|
| `MRegularSeasonDetailedResults.csv` | Men's regular season box scores → Sabermetrics, Elo |
| `WRegularSeasonDetailedResults.csv` | Women's regular season box scores → Sabermetrics, Elo |
| `MNCAATourneyDetailedResults.csv` | Men's tournament results → Training labels |
| `WNCAATourneyDetailedResults.csv` | Women's tournament results → Training labels |
| `MNCAATourneySeeds.csv` | Men's tournament seeds → Seed feature |
| `WNCAATourneySeeds.csv` | Women's tournament seeds → Seed feature |
| `MNCAATourneyCompactResults.csv` | Men's tournament compact → Post-hoc evaluation |
| `WNCAATourneyCompactResults.csv` | Women's tournament compact → Post-hoc evaluation |
| `SampleSubmissionStage2.csv` | Submission template (519,144 rows) |

### External Datasets Integration

**No external datasets were used.** All features are derived solely from the official competition data.

### Missing Data Imputation Techniques

- **Quality scores**: If a team has no GLM Quality score for a given season (e.g., insufficient games), it is filled with the column mean.
- **Elo scores**: Missing Elo values are filled with the baseline rating of 1500.
- **Seed values**: Teams without tournament seeds (non-tournament teams appearing in submission pairs) receive `NaN` seeds, which are filled with 0 during Delta computation.
- **Delta features**: Any missing Delta values during inference are filled with 0 (neutral assumption).

### Outlier Detection and Handling

Outliers are **intentionally preserved**. March Madness upset games produce extreme statistical values that are legitimate signals, not noise. The `RobustScaler` (which uses median and interquartile range instead of mean and standard deviation) provides natural resilience to outliers without removing them.

### Data Consistency & Normalization

- **Time-Travel Filter**: All data for Season 2026 is truncated at DayNum ≤ 118. Seeds for 2026 are completely removed.
- **Winner/Loser relabeling**: Raw columns prefixed with `W` (winner) and `L` (loser) are re-mapped to neutral `Team` / `Opponent` labels before aggregation, so the same team's stats are consistently grouped regardless of game outcome.
- **Scaling**: `RobustScaler` is fitted on training data and applied consistently to both validation and test data.

---

## 4. Feature Engineering

### Historical Performance Metrics

From aggregated Regular Season box scores, the following per-game averages are computed:

| Feature | Description |
|---------|-------------|
| `wpct` | Win percentage |
| `margin` | Average point margin per game |
| `stlpg` | Steals per game |
| `blkpg` | Blocks per game |
| `drbpg` | Defensive rebounds per game |

### Efficiency Ratings (Offensive/Defensive)

All efficiency metrics are normalized per 100 possessions, where possessions are estimated as:

```
poss = FGA − ORB + TO + 0.475 × FTA
```

| Feature | Formula | Description |
|---------|---------|-------------|
| `oeff` | pts / poss × 100 | Offensive efficiency |
| `deff` | ops / oposs × 100 | Defensive efficiency |
| `neff` | oeff − deff | Net efficiency differential |
| `efg` | (FGM + 0.5×F3M) / FGA | Effective field goal % |
| `oefg` | (oFGM + 0.5×oF3M) / oFGA | Opponent effective FG% |
| `tor` | TO / poss | Turnover rate |
| `otor` | oTO / oposs | Opponent turnover rate |
| `orpct` | ORB / (ORB + oDRB) | Offensive rebound % |
| `oorpct` | oORB / (oORB + DRB) | Opponent offensive rebound % |
| `ftr` | FTM / FGA | Free throw rate |
| `oftr` | oFTM / oFGA | Opponent free throw rate |
| `f3pct` | F3M / F3A | 3-point field goal % |
| `of3pct` | oF3M / oF3A | Opponent 3-point FG% |
| `astr` | AST / FGM | Assist ratio |
| `pace` | (poss + oposs) / (2 × n) | Game pace (possessions per game) |

> **Important:** None of these 20 advanced stats exist in the raw CSV files. They are all **derived** from the box-score columns (`WFGM`, `WFGA`, `WFGM3`, `WFGA3`, `WFTM`, `WFTA`, `WOR`, `WDR`, `WAst`, `WTO`, `WStl`, `WBlk`, `WPF`, `LFGM`, `LFGA`, `LFGM3`, `LFGA3`, `LFTM`, `LFTA`, `LOR`, `LDR`, `LAst`, `LTO`, `LStl`, `LBlk`, `LPF`, plus `WScore`, `LScore`).

### Strength of Schedule (SOS) Adjustments

SOS is implicitly captured through two mechanisms:
- **Elo Rating**: Teams that defeat strong opponents (high Elo) gain more rating points, naturally encoding schedule strength.
- **GLM Quality**: The GLM decomposes point differentials into team-level coefficients, inherently adjusting for opponent strength.

No explicit SOS feature is computed separately.

### Seed and Ranking Features

| Feature | Description |
|---------|-------------|
| `Delta_Seed` | `Seed_TeamA − Seed_TeamB` (lower seed = stronger) |
| `T1_Seed` / `T2_Seed` | Individual seed values as context features |
| `T1_Elo` / `T2_Elo` | Individual Elo ratings as context features |
| `T1_Quality` / `T2_Quality` | Individual Quality scores as context features |

### Interaction and Difference Features (Matchup Logic)

All 22 team-level features are converted into **Delta (difference)** features:

```
Delta_X = Stat_TeamA − Stat_TeamB
```

This matchup-relative representation has two key advantages:
1. The model learns *relative differences* rather than absolute values, which transfers better across eras.
2. Combined with symmetric augmentation, it enforces order-invariance.

### Feature Selection and Dimensionality Reduction

- Features with `|Pearson correlation with Target| < 0.05` are removed as noise.
- Typical noise features: `Delta_otor`, `Delta_pace`, `Delta_oorpct`.
- No PCA or other dimensionality reduction is applied — the feature space (~20 Delta features + context features) is small enough that GBDT models handle it efficiently.

---

## 5. Model Architecture

### Base Learners

| Model | Objective | Role |
|-------|-----------|------|
| XGBoost (`XGBRegressor`) | `reg:squarederror` | Base learner 1 |
| LightGBM (`LGBMRegressor`) | `regression` | Base learner 2 |

Both models are trained separately on Men's and Women's data, yielding 4 trained models total:
- `model_xgb_m`, `model_lgb_m` (Men's)
- `model_xgb_w`, `model_lgb_w` (Women's)

### Neural Network Architectures

Not applicable — no neural networks are used in this solution.

### Custom Model Components

- **Elo Rating Engine** (`calculate_elo_fast`): Custom iterative algorithm with Mean Reversion, Home Court Advantage, and Logarithmic MOV scaling.
- **GLM Quality Engine** (`calculate_glm_quality`): Uses `statsmodels.GLM` with Gaussian family to extract per-team strength coefficients season-by-season.

### Ensemble Structure

**Simple Averaging** — the final prediction is the arithmetic mean of XGBoost and LightGBM predictions:

```python
pred_ensemble = (pred_xgb + pred_lgb) / 2
```

No stacking or blending meta-learner is used. This deliberate simplicity avoids overfitting on the small training set.

---

## 6. Training & Optimization

### Objective and Loss Functions

| Model | Objective | Loss |
|-------|-----------|------|
| XGBoost | `reg:squarederror` | Mean Squared Error |
| LightGBM | `regression` | Mean Squared Error |

MSE is chosen because it directly aligns with the Brier Score metric used for evaluation.

### Cross-Validation Strategy

- **Method**: `GroupKFold` from scikit-learn.
- **Groups**: `Season` column — each fold holds out one or more **complete seasons** as validation.
- **Rationale**: Prevents within-season data leakage and tests the model's ability to generalize to future unseen seasons.

### Early Stopping Criteria

- XGBoost: `n_estimators=1000`
- LightGBM: `n_estimators=1000`

### Hyper-parameter Tuning

Hyper-parameters were manually tuned. See Section 7 for the full parameter list.

### Hardware and Runtime Environment

| Item | Value |
|------|-------|
| Platform | Kaggle Notebooks / Google Colab |
| GPU | `None` |
| Python Version | 3.12 |

---

## 7. Hyper-parameters List

### Global Parameters

| Parameter | Value |
|-----------|-------|
| Time-Decay λ | 0.15 |
| Elo K-factor | 20 |
| Elo Home Court Advantage | 100 |
| Elo Mean Reversion Factor | 0.75 |
| Noise Threshold (correlation) | 0.05 |
| Validation Clip Range | [0.0025, 0.9975] |
| Inference Clip Range | [0.02, 0.98] |

### Model-Specific Parameters

#### XGBoost (`XGBRegressor`)

| Parameter | Value |
|-----------|-------|
| `objective` | `reg:squarederror` |
| `n_estimators` | 1000 |
| `max_depth` | `5` |
| `learning_rate` | `0.01` |
| `subsample` | `0.8` |
| `colsample_bytree` | `0.8` |
| `min_child_weight` | `15` |
| `random_state` | `42` |

#### LightGBM (`LGBMRegressor`)

| Parameter | Value |
|-----------|-------|
| `objective` | `regression` |
| `metric` | `mse` |
| `n_estimators` | 1000 |
| `max_depth` | `4` |
| `learning_rate` | `0.01` |
| `num_leaves` | `10` |
| `random_state` | `42` |
| `verbose` | `-1` |

### Learning Rates and Regularization Settings

- **Learning Rate Strategy:** A low learning rate of 0.01 was chosen for both XGBoost and LightGBM. This ensures a slow, incremental learning process, reducing the risk of overshooting the global minimum in a noisy environment.

- **Tree Constraints:** `Maximum Depth` restricted to 4 (LGBM) and 5 (XGB) to prevent the models from capturing complex, non-generalizable noise in specific matchups.

- **Leaf/Weight Constraints:** Used `min_child_weight=15` in XGBoost and `num_leaves=10` in LightGBM to ensure each leaf node represents a significant enough sample size to be statistically meaningful.

- **Stochastic Regularization:** `Subsample` & `Colsample` = 0.8 was applied to both row and column sampling per tree, ensuring the ensemble remains robust to specific feature outliers or individual dominant seasons.

---

## 8. Post-Processing & Submission

### Probability Clipping/Smoothing

Two-stage clipping is applied:

| Stage | Range | Purpose |
|-------|-------|---------|
| Validation (internal eval) | [0.0025, 0.9975] | Prevent extreme Brier penalties during CV evaluation |
| Inference (final submission) | [0.02, 0.98] | Conservative bounds for the actual Kaggle submission |

No isotonic regression, Platt scaling, or other calibration methods are applied in the final pipeline.

### Upset Adjustment Strategies

No explicit upset adjustment is applied. The clipping bounds serve as an implicit upset hedge — the model can never predict a team wins with more than 98% probability, providing a safety margin against the quadratic penalty of Brier Score on upsets.

### Submission Formatting

- Men's (`TeamID < 2000`) and Women's (`TeamID ≥ 2000`) predictions are generated separately then merged.
- Output format: `ID` (format: `Season_TeamA_TeamB`) and `Pred` (win probability for TeamA).
- Total rows: 519,144 (matching `SampleSubmissionStage2.csv` exactly).
- File: `submission.csv`

---

## 9. Code Repository & Reproducibility

### Repository Link

| Platform | Link |
|----------|------|
| Kaggle Notebook |  [https://www.kaggle.com/code/huphcphmquang/8th-place-march-machine-learning-mania-2026](https://www.kaggle.com/code/huphcphmquang/8th-place-march-machine-learning-mania-2026) |
| GitHub | [https://github.com/huuphuoc-phamquang/8th-march_ML_mania](https://github.com/huuphuoc-phamquang/8th-march_ML_mania) |

### Environment Setup

**Required packages** (Python 3.12):

```
pandas
numpy
matplotlib
seaborn
statsmodels
scikit-learn
xgboost
lightgbm
tqdm
```

### Directory Structure and Data Placement

```
project/
├── notebook.ipynb           # Main notebook
├── Solution_Writeup.md   # This document
├── submission.csv        # Generated output
├── stage1/               # Old version of the competition data
│   ├── MRegularSeasonDetailedResults.csv
│   └── WRegularSeasonDetailedResults.csv
├── stage2/               # New version of the competition data
│   ├── MNCAATourneyDetailedResults.csv
│   ├── WNCAATourneyDetailedResults.csv
│   ├── MNCAATourneySeeds.csv
│   ├── WNCAATourneySeeds.csv
│   ├── MNCAATourneyCompactResults.csv
│   ├── WNCAATourneyCompactResults.csv
│   └── SampleSubmissionStage2.csv
```

> **Note:** The `DATA_PATH` and `DATA_PATH2` variables in the notebook must be updated to point to the correct data directory on your system.

### Step-by-Step Execution Guide

| Step | Notebook Cell | Description |
|------|---------------|-------------|
| 0 | Step 0 | Import all dependencies |
| 1 | Step 1 | Load raw CSV data |
| 2 | Step 2 | Apply Time-Travel Filter |
| 3 | Step 3 | Define Time-Decay weight function |
| 4 | Step 4 | Compute Elo ratings |
| 5 | Step 5 | Compute GLM Quality ratings |
| 6 | Step 6 | Compute Sabermetrics & build Team Profiles |
| 7 | Step 7 | Build Matchup training dataset |
| 8 | Step 8 | Feature analysis & noise filtering |
| 9 | Step 9 | Train XGBoost + LightGBM ensemble |
| 10 | Step 10 | Clip validation predictions |
| 11 | Step 11 | Generate final submission file |

**Total estimated runtime:** ~10-15 minutes on a standard CPU machine.

### Hardware Dependencies and Execution Time

- **No GPU required.** The entire pipeline runs on CPU.
- Tested on: Kaggle Notebooks (4 CPU cores, 30GB RAM) and Google Colab (free tier).
- The most time-consuming step is GLM Quality computation (~5-10 minutes) due to fitting a separate Gaussian GLM for each of the 24 (Men's) + 17 (Women's) seasons.

---

## 10. Results & Insights

### Feature Importance Analysis

Based on XGBoost's feature importance (information gain), the most impactful features are typically:

| Rank | Feature | Importance |
|------|---------|------------|
| 1 | `Delta_Quality` | `0.205751` |
| 2 | `Delta_Seed` | `0.090144` |
| 3 | `Delta_Elo` | `0.079722` |
| 4 | `Delta_wpct` | `0.036574` |
| 5 | `Delta_deff` | `0.034711` |

### Local vs. Leaderboard Correlation

| Metric | All times | From 2015 | 2026 |
|--------|----------|-----------|------------|
| Brier Score (Men) | 0.19138 | 0.18807 | `—` |
| Brier Score (Women) | 0.14611 | 0.14815 | `—` |
| Brier Score (Combined) | `—` | `—` | 0.11845 |

### What Worked and What Failed (Ablation Study)

**What Worked:**

| Technique | Effect |
|-----------|--------|
| Time-Travel Filter | Essential — without it, local CV scores are misleadingly good |
| Dual-Rating System (Elo + Quality) | Better than either alone |
| RobustScaler over StandardScaler | Small but consistent improvement |
| Probability Clipping | Significant protection against upset penalties |
| Time-Decay Weighting | Modest improvement by de-emphasizing old seasons |

**What Did Not Work / Was Not Tried:**

| Technique | Outcome |
|-----------|---------|
| Platt Scaling / Isotonic Calibration | `Tried but underperformed` |
| External data (e.g., AP polls, KenPom) | `Not used, multicollinearity` |

---

## Conclusion

### Technical Summary

This solution demonstrates that a well-engineered feature pipeline combined with a simple but robust ensemble can produce competitive March Madness predictions. The key insight is that **respecting the nature of the data** (preserving outliers, using robust scaling, clipping extreme probabilities) matter more than model complexity.

The Dual-Rating System (Elo for historical strength + GLM Quality for current-season form) provides complementary signals that a single rating system cannot capture alone. Combined with 20 derived Sabermetrics, the model gets a comprehensive multi-dimensional profile of each team, enabling effective matchup-level predictions.

### Critical Reflection — A Structural Vulnerability of the Competition

Perhaps the most revealing finding of this project is not a technical one, but a methodological one: **this solution achieved a Top 8 finish while using outdated data.**

As documented in Section 1 and Section 3, the `RegularSeasonDetailedResults` files used in this pipeline come from an older Kaggle data release (`stage1`). The 2026 regular-season data is truncated at DayNum 118, missing all late-season and conference tournament games. Some box-score statistics for prior seasons may also be stale. Despite this, the pipeline produced a Brier Score competitive enough to rank in the top 8 out of all participating teams.

### Future Improvements

- Automated hyper-parameter tuning (e.g., Optuna).
- Additional ensemble members (CatBoost, neural networks).
- Incorporating conference tournament results as late-season signal.
- More sophisticated calibration techniques (e.g., Venn-Abers prediction).
- **Most importantly**: updating to the latest Kaggle data release to test whether current-season completeness actually improves predictions — or confirms that stale data is "good enough."
