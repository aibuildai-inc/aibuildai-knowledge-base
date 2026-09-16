# 2nd Place Solution for the March Machine Learning Mania 2026 Competition

Competition: march-machine-learning-mania-2026
Rank: #2
Source: https://www.kaggle.com/c/march-machine-learning-mania-2026/writeups/2nd-place-solution-for-the-march-machine-learning

**Final result:** 2nd place | Brier score: **0.1149886** | **Rank:** 2nd/3,485 teams

## Context

**Business context:** https://www.kaggle.com/competitions/march-machine-learning-mania-2026/overview

**Data context:** https://www.kaggle.com/competitions/march-machine-learning-mania-2026/data



## Overview of the Approach

### Models and Algorithms

The final submission is a weighted ensemble of two models: **XGBoost (60%) and Logistic Regression (40%)**, trained on historical NCAA tournament results (men's back to 2003, women's to 2010) combined into a single training set.

```python
LR_WEIGHT = 0.4
XGB_WEIGHT = 0.6

blended = LR_WEIGHT * lr_preds + XGB_WEIGHT * xgb_preds
blended = np.clip(blended, 0.02, 0.98)
```

XGBoost captures non-linear feature interactions. Logistic Regression contributes calibrated probability estimates that reduce overconfidence at the extremes. The prediction clip to [0.02, 0.98] is a deliberate Brier-score-aware decision: a 0.99 wrong prediction costs (0.99)² = 0.9801 in Brier loss; at 0.98 the cost is 0.9604. Across 63+ games this compounds meaningfully.

One important implementation detail: **LR is trained on StandardScaler-normalized features; XGBoost is trained on raw features.** Both receive the same feature vector, but the pipelines are separate.

```python
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_train)

lr = LogisticRegression(C=1.0, max_iter=1000, solver='lbfgs')
lr.fit(X_scaled, y_train)

xgb_model = xgb.XGBClassifier(
    n_estimators=300, max_depth=4, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
    reg_alpha=0.1, reg_lambda=1.0, eval_metric='logloss',
    use_label_encoder=False, random_state=42
)
xgb_model.fit(X_train, y_train)  # raw, unscaled
```

At inference, both models receive the same matchup features via batch prediction:

```python
X_s = scaler.transform(X)           # scaled for LR
lr_preds = lr.predict_proba(X_s)[:, 1]
xgb_preds = xgb_model.predict_proba(X)[:, 1]  # raw for XGB

blended = LR_WEIGHT * lr_preds + XGB_WEIGHT * xgb_preds
blended = np.clip(blended, 0.02, 0.98)
```

Any matchup where either team's profile is missing defaults to 0.5.

### Data Preprocessing and Feature Engineering

All 35 features are computed as **Team1 − Team2 differences**, where Team1 is always the lower TeamID. Framing features as relative rather than absolute forces the model to learn what separates teams in a matchup and generalizes better across eras and between genders.

All features are derived from Kaggle-provided datasets. No external data was used.

**Processing pipeline:**

**Season stats** are computed from `MRegularSeasonDetailedResults` / `WRegularSeasonDetailedResults` and `MRegularSeasonCompactResults` / `WRegularSeasonCompactResults`. For each team-season, the pipeline computes: win percentage, point differential, average points scored and allowed, and — from detailed results — offensive/defensive/net efficiency per 100 possessions (using estimated possessions: `FGA − OR + TO + 0.475 × FTA`), Four Factors (EFG%, turnover rate, offensive rebounding percentage, free throw rate), shooting splits (FG%, FG3%, FT%), opponent FG%, per-game assists/turnovers/steals/blocks/offensive rebounds/defensive rebounds, and tempo.

```python
dg['Poss'] = dg['FGA'] - dg['OR'] + dg['TO'] + 0.475 * dg['FTA']
adv['OffEff'] = (adv['TotalPts'] / adv['Poss'].replace(0, 1)) * 100
adv['DefEff'] = (adv['TotalOppPts'] / adv['OppPoss'].replace(0, 1)) * 100
adv['NetEff'] = adv['OffEff'] - adv['DefEff']
adv['EFGPct'] = (adv['FGM'] + 0.5 * adv['FGM3']) / adv['FGA'].replace(0, 1)
adv['TORate'] = adv['TO'] / adv['Poss'].replace(0, 1)
adv['ORPct'] = adv['OR'] / (adv['OR'] + adv['OppDR']).replace(0, 1)
```

**Elo ratings** are computed fresh each season from regular season results, with K=20 and 75% mean reversion to 1500 at the start of each new season. The K-factor is further scaled by a **margin-of-victory multiplier** that rewards decisive wins while correcting for the fact that large margins are more likely between unequal teams:

```python
def compute_elo(compact_df, k=20, home_adv=100, mean_reversion=0.75):
    elo = {}
    for season in sorted(compact_df['Season'].unique()):
        # Mean reversion at season start
        for team in elo:
            elo[team] = elo[team] * mean_reversion + 1500 * (1 - mean_reversion)
        for _, game in compact_df[compact_df['Season'] == season].sort_values('DayNum').iterrows():
            w, l = game['WTeamID'], game['LTeamID']
            if w not in elo: elo[w] = 1500
            if l not in elo: elo[l] = 1500
            w_elo, l_elo = elo[w], elo[l]
            if game['WLoc'] == 'H': w_elo += home_adv
            elif game['WLoc'] == 'A': l_elo += home_adv
            w_exp = 1 / (1 + 10 ** ((l_elo - w_elo) / 400))
            mov = game['WScore'] - game['LScore']
            mov_mult = np.log(max(mov, 1) + 1) * (2.2 / ((w_elo - l_elo) * 0.001 + 2.2))
            elo[w] += k * mov_mult * (1 - w_exp)
            elo[l] -= k * mov_mult * (1 - w_exp)
```

**Massey ordinals** are filtered to the last 2 weeks of the regular season from the provided `MMasseyOrdinals` table, with three aggregates per team: mean, median, and minimum rank across all available systems. **Note: Massey ordinals are available only for men's teams in the competition dataset.** Women's teams receive a default of 200 (below the typical ranking range) for all three Massey features.

```python
def get_massey(ordinals_df, season):
    so = ordinals_df[ordinals_df['Season'] == season]
    max_day = so['RankingDayNum'].max()
    late = so[so['RankingDayNum'] >= max_day - 14]
    return late.groupby('TeamID')['OrdinalRank'].agg(
        MasseyMean='mean', MasseyMedian='median', MasseyMin='min'
    ).reset_index()
```

**Seeds** are extracted from `MNCAATourneySeeds` / `WNCAATourneySeeds` as raw integer values (e.g., seed "W04" → 4). Unseeded teams get a default of 17.

**Strength of schedule** is the average win percentage of all opponents faced during the regular season.

**Recent form** is last-10-game win percentage.

All 35 matchup features are differences (Team1 − Team2):

| Group | Features |
|---|---|
| Basic season stats | WinPct, PointDiff, AvgPts, AvgOppPts |
| Advanced efficiency | OffEff, DefEff, NetEff (per 100 possessions) |
| Four Factors | EFGPct, TORate, ORPct, FTRate |
| Shooting splits | FGPct, FG3Pct, FTPct, OppFGPct |
| Per-game counting stats | AstPerGame, TOPerGame, StlPerGame, BlkPerGame, ORPerGame, DRPerGame, Tempo |
| Opponent Four Factors | OppEFGPct, OppTORate, DRPct, OppFTRate |
| Elo | EloDiff |
| Seeds | SeedDiff, Seed1, Seed2 |
| Massey ordinals (men's only) | MasseyMeanDiff, MasseyMedianDiff, MasseyMinDiff |
| Strength of schedule | SOSDiff |
| Recent form | RecentFormDiff |

### Validation Strategy

10-fold cross-validation on `neg_brier_score`, run on the combined men's + women's training set before the final full-data fit:

```python
from sklearn.model_selection import cross_val_score

lr_brier = -cross_val_score(lr, X_scaled, y_train, cv=10, scoring='neg_brier_score').mean()
xgb_brier = -cross_val_score(xgb_model, X_train, y_train, cv=10, scoring='neg_brier_score').mean()
```

| Model | Men's CV Brier | Women's CV Brier |
|---|---|---|
| Logistic Regression | 0.1903 ± 0.0159 | 0.1425 ± 0.0111 |
| XGBoost | 0.2009 ± 0.0170 | 0.1552 ± 0.0180 |
| Blended (40% LR + 60% XGB) | 0.1925 ± 0.0162 | 0.1480 ± 0.0151 |

LR outperformed XGBoost standalone in both genders on CV. The blend's value is XGBoost adding non-linear interactions that LR can't capture, not raw individual performance.

> **A note on the CV–competition gap:** Cross-validation estimated a blended Brier of ~0.192 (men's) and ~0.148 (women's). The final competition score of 0.1149886 outperformed these estimates considerably. This gap does not imply model superiority; rather, it reflects two compounding realities of the tournament:
>
> 1. **Small Sample Variance:** With only ~63 scored games per gender, a few predictable matchups heavily skew the final metric.
>
> 2. **A "Chalky" Year:** 2026 appears to have been a historically low-upset tournament. This natively rewards conservative models that correctly assign high probabilities to heavy favorites.
>
> Ultimately, the CV figures remain the more reliable estimate of what this ensemble will produce in an average tournament year.



## Details of the Submission

### What Was Important and Impactful

**Difference-vector framing.** Every feature is Team1 minus Team2. This forces the model to learn what separates teams in a matchup rather than overfit to team-specific patterns that don't generalize across eras. It also makes the combined men's/women's training set work cleanly, since relative-strength dynamics are gender-agnostic. Doubling the effective training set meaningfully helps calibration on rare events.

**Massey ordinal consensus.** The strongest non-seed signal for men's games. Aggregating 50+ independent ranking systems into mean/median/minimum gives a consensus that's more robust than any single metric. Filtering to the last two weeks of the regular season specifically captures teams peaking into tournament time.

**Margin-of-victory Elo.** The Elo implementation goes beyond standard K-factor updates by incorporating a margin-of-victory multiplier (`log(MOV+1)`) with an autocorrelation correction that prevents runaway updates when already-dominant teams win big. This gives the Elo feature more predictive texture than a standard binary win/loss update.

**Separate scaling per model.** LR receives StandardScaler-normalized features; XGBoost receives raw features. This matters because LR's coefficient learning is sensitive to feature scale while tree-based models are not — using unscaled features for LR would distort the regularization (C=1.0) applied to each coefficient.

**Prediction clipping at [0.02, 0.98].** A deliberate Brier-score-aware choice. At 0.99 wrong, Brier loss is 0.9801; at 0.98 wrong it's 0.9604. Across 63+ games this difference compounds meaningfully.


### What Didn't Work

**Deeper XGBoost trees**: `max_depth=6` and `max_depth=8` both showed worse CV Brier scores — clear overfitting on ~1,000 total tournament games. `max_depth=4` was the ceiling before degradation.

**60/40 blend ratio was empirical, not grid-searched.** A proper grid search over blend weights with nested CV would have been cleaner. 

**No Massey ordinals for women's teams.** The `MMasseyOrdinals` table in the competition data covers men's teams only. Women's teams fall back to a default of 200 for all Massey features, which is a meaningful information gap. Finding or constructing an equivalent women's composite ranking is a clear improvement opportunity.

**No upset-specific seed-matchup features.** Historical upset rates by seed pairing (12-seeds vs 5-seeds run ~35%, 11-seeds run ~37%) are stable signals that the current feature set doesn't explicitly capture. The model picks these up implicitly through seeds and Massey, but not through a direct prior.

**Single combined model.** Training men's and women's data together is efficient but sacrifices the ability to tune each tournament's dynamics independently. Separate models with a calibrated ensemble blend might improve both genders.



## Key Hyperparameter Decisions

| Decision | Value | Rationale |
|---|---|---|
| XGB/LR blend | 60/40 | Empirical; LR provides calibration, XGB carries predictive weight |
| XGB max_depth | 4 | Prevents overfitting on small tournament dataset (~1K games) |
| XGB n_estimators | 300 | With lr=0.05, sufficient rounds to converge |
| XGB reg_alpha / reg_lambda | 0.1 / 1.0 | L1+L2 regularization to generalize on small dataset |
| LR C | 1.0 | Standard inverse regularization strength |
| Elo K-factor | 20 | Base; effective K scaled up by MOV multiplier |
| Elo mean reversion | 75% toward 1500 | Balances historical signal vs. recency |
| Probability clip | [0.02, 0.98] | Directly minimizes Brier loss at tails |
| Massey window | Last 2 weeks of regular season | Most predictive period for tournament performance |
| CV folds | 10 | Stable Brier estimate on small dataset |
| Default unseeded SeedNum | 17 | One beyond max seed (16) for teams not in tournament |



## External Data

None. All features were derived from the datasets provided in the competition. Note that Massey ordinals (included in the competition data) cover men's teams only; women's teams received a fixed default value for those features.



## Open Source / Tooling

- **XGBoost** (`xgboost` Python library)
- **scikit-learn** (`LogisticRegression`, `StandardScaler`, `KFold`, `cross_val_score`)
- **pandas / numpy** for data processing


## What I'd Try Next

1. **Women's composite rankings.** The absence of Massey ordinals for women is the biggest addressable gap. Building or sourcing an equivalent consensus ranking (e.g., from ESPN BPI, Her Hoop Stats, or NCAA's own NET rankings) would likely improve women's calibration meaningfully.
2. **Isotonic regression calibration** post-blend. Current calibration relies on LR's linear structure — explicit post-hoc calibration would be cleaner.
3. **Grid search over blend weights** with nested CV rather than empirical selection.
4. **Seed × Massey interaction features**: a team with high consensus ranking and a favorable seed (classic dangerous mid-seed) is currently captured implicitly but not explicitly.
5. **Separate men's and women's models** with a post-hoc calibrated blend.
6. **Tournament-trajectory features**: does a team's path through the bracket affect later-round performance? Hard to measure cleanly but worth exploring.


## Steps to Reproduce

### Requirements

- Python 3.9+
- ~4 GB RAM
- ~5 minutes runtime

### 1. Clone the repository

```bash
git clone https://github.com/BrendanCarlin/march-mania-2026-2nd-place.git
cd march-mania-2026-2nd-place
```

### 2. Set up the environment

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
```

### 3. Download competition data

Download all data files from the [Kaggle competition data page](https://www.kaggle.com/competitions/march-machine-learning-mania-2026/data) and place them in a `data/` directory at the repo root. The script requires these files:

```
data/
├── MRegularSeasonDetailedResults.csv
├── MRegularSeasonCompactResults.csv
├── MNCAATourneyCompactResults.csv
├── MNCAATourneySeeds.csv
├── MTeams.csv
├── MMasseyOrdinals.csv
├── WRegularSeasonDetailedResults.csv
├── WRegularSeasonCompactResults.csv
├── WNCAATourneyCompactResults.csv
├── WNCAATourneySeeds.csv
├── WTeams.csv
└── SampleSubmissionStage2.csv
```

### 4. Create the output directory

```bash
mkdir output
```

### 5. Generate the submission

```bash
python march-mania-2026.py
```

Output: `output/submission.csv` — 132,133 rows of win probability predictions for all possible 2026 matchups.

The script is fully deterministic (`random_state=42` on XGBoost; no stochastic elements in LR). Re-running produces a bit-for-bit identical submission to the one that placed 2nd (verified by diff).



## Sources

- **XGBoost**: Chen, T., & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. *KDD 2016*. https://arxiv.org/abs/1603.02754
- **Margin-of-victory Elo**: FiveThirtyEight NFL Elo methodology (the MOV multiplier approach). https://fivethirtyeight.com/methodology/how-our-nfl-predictions-work/
- **Massey Ordinals**: Kenneth Massey's Composite Rankings. https://masseyratings.com
- **Brier Score**: Brier, G.W. (1950). Verification of forecasts expressed in terms of probability. *Monthly Weather Review*.
- **Four Factors**: Oliver, D. (2004). *Basketball on Paper*. The foundational reference for EFG%, TORate, ORPct, FTRate.
- **scikit-learn documentation**: https://scikit-learn.org/stable/
- **Competition discussion forum**: https://www.kaggle.com/competitions/march-machine-learning-mania-2026/discussion
