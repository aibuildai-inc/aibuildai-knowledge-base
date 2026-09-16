# 7th Place Solution - abobus team

Competition: march-machine-learning-mania-2026
Rank: #7
Source: https://www.kaggle.com/c/march-machine-learning-mania-2026/writeups/7th-place-solution

We want to start by thanking @asimandia and @raddar for the public baseline that we built our solution on. It gave us a clean and sensible starting point that has already become a classic backbone of any successful March Mania solution. We kept that overall structure, trusted the parts that already made sense, and spent our time improving the pieces that looked easiest to move.

At a higher level, our final solution had three main parts:

- a richer set of matchup features built from seeds, Elo, GLM team quality, and possession-based efficiency metrics
- an XGBoost regressor trained on point differential values rather than a binary win/loss target
- separate men's and women's isotonic calibration to improve robustness, followed by light sharpening learned from OOF predictions.

Eventually, we stayed close to the baseline, but ended up with a stronger final pipeline.

## Baseline and what we changed

Pizzaboi and Raddar's baseline already had a lot of advanced sports and ML related techniques included. Tournament games were converted into a Team 1 vs Team 2 table, regular-season summaries were merged in, Elo and a GLM-style quality estimate were added to the data, and an XGBoost regressor was trained on point differential.

Most of our gains came from tightening the following four areas:

### 1) We made the feature set richer

The baseline already used seeds, regular-season averages, Elo, and GLM quality. We kept all of that and added several more layers. This was the biggest step forward over the baseline. The final model used **89 engineered features**, and the feature importance plot below shows that the extra quality and efficiency gaps became central.

### 2) We normalized game statistics more carefully

We also normalized overtime-affected box score statistics back toward a 40-minute baseline before building season summaries. That sounds minor, but once you start leaning on possession-based and efficiency-based features, those small distortions matter. We found it cleaner to standardize those inputs up front rather than let the model absorb overtime noise.

### 3) We replaced the baseline probability mapping with a more deliberate calibration pipeline

The baseline used a simpler margin-to-probability conversion. We kept the margin-first idea, but made the probability side more careful:

- generate leave-one-season-out margin predictions;
- fit **separate isotonic regressions** for men's and women's games;
- apply a small amount of **tail sharpening** to extreme probabilities, chosen on out-of-fold data.

In our opinion, this was one of the most useful changes. The men's and women's tournaments do not behave exactly the same, and treating them separately gave us better estimates.

### 4) We kept validation and submission as close as possible

For submission, we averaged margin predictions from the seasonal models and then applied the same group-specific calibration logic we used during validation. That may sound obvious, but it helped avoid the usual trap where the cross-validation pipeline and the final submission pipeline quietly drift apart.

## Data setup

We trained on the official competition files for both tournaments. Each historical tournament game was represented from the Team 1 vs Team 2 perspective with:

- `PointDiff = T1_Score - T2_Score` as the regression target;
- `win = 1(PointDiff > 0)` as the evaluation label for probability metrics.

We merged tournament results with team-level regular-season features, seeds, Elo ratings, and GLM quality estimates, then built matchup-level predictors from both absolute team strength and team-vs-team differences.

## Feature engineering

The features that we used in the gradient boosting model can be grouped into six main groups:

| Feature group                        | Features                                                                                                                                | Why it helped                                                                                                                                           |
| ------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Seed features**                    | Raw seeds, `Seed_diff`, `Seed_sum`, `Seed_prod`, `Seed_gap_abs`                                                                         | These let the model go beyond simple seed gap and distinguish, for example, a strong-bracket game from a lower-seeded matchup with the same difference. |
| **Elo features**                     | `elo_diff`, `elo_prob`                                                                                                                  | We found it useful to keep both the raw rating gap and its implied win probability. `elo_prob` gave the trees a smoother, bounded strength signal.      |
| **Possession / efficiency features** | Possessions, `OffRtg`, `DefRtg`, `NetRtg`, `eFG%`, `TS%`, `FTR`, 3PA rate, turnover rate, rebounding rates, assist rate, win percentage | These features described *how* teams were winning: shot quality, pace, ball security, rebounding, and overall efficiency.                               |
| **Adjusted strength features**       | `AdjOffRtg`, `AdjDefRtg`, `AdjNetRtg`, `PythagWinPct`, `Luck`                                                                           | These corrected for schedule strength and gave us cleaner estimates of underlying team quality than raw season averages alone.                          |
| **Matchup gap features**             | Differences for most advanced stats, e.g. `AdjNetRtg_diff`, `OffRtg_diff`, `TS_diff`, `ORBRate_diff`                                    | Tournament prediction is mostly about relative strength, so these features let the model reason directly about the gap between the two teams.           |
| **Blended prior**                    | `strength_blend = 0.45 * diff_quality + 0.35 * elo_diff + 0.20 * AdjNetRtg_diff`                                                        | This gave the model one strong summary feature combining three of our most reliable views of team strength.                                             |

[Top XGBoost features]
*Figure 1. Quality difference, adjusted efficiency difference, and seed-gap variables carried most of the signal.*

## Model and validation

Our core model was an **XGBoost regressor** with `objective = reg:squarederror`. We kept the tree model fairly conservative: `eta = 0.0093`, `max_depth = 4`, `min_child_weight = 4`, `subsample = 0.6`, `colsample_bynode = 0.8`, `num_parallel_tree = 2`, `grow_policy = lossguide`, `max_bin = 38`, and `num_boost_round = 704`.

We validated with **leave-one-season-out cross-validation**. For each held-out season, we trained on all other seasons, predicted tournament margins for the held-out year, and then ran the calibration step on the out-of-fold predictions.

The notebook reported:
- average leave-one-season-out margin **MAE = 9.2418**;
- overall calibrated out-of-fold **Brier = 0.16272**;
- best displayed recent fold: **2025 Brier = 0.12803**.

Cross-validation was reasonably steady from season to season, which gave us confidence that the changes were helping for the right reasons, even if the final public/private shakeout still depended on bracket variance.

## Key metrics

| Metric | Value |
|---|---:|
| Seed-only AUC | 0.811 |
| GLM team-quality AUC | 0.829 |
| Average leave-one-season-out MAE | 9.2418 |
| Overall calibrated OOF Brier | 0.16272 |
| Best reported 2025 fold Brier | 0.12803 |

## Submission pipeline

For Stage 2, we rebuilt the same matchup features for every row in `SampleSubmissionStage2.csv`, scored them with the seasonal XGBoost models, averaged the predicted margins, and then applied the learned men's and women's calibrators. Final probabilities were clipped to `[0.01, 0.99]` and rounded before export.

## Final thoughts

We built our solution on top of **Pizzaboi** and **Raddar's** public baseline, so first of all, thank you to them for sharing such a strong starting point. The baseline already had the core idea in place; our contribution was to push it further by improving the team-strength features, making the matchup representation more expressive, and putting more care into calibration.

We also want to thank the **Kaggle team** for continuing to host March Mania. It is one of the most fun competitions of the year, but it is also one where variance matters a lot. In this setting, the gap between a solid result and a top-8 finish is never just model quality. A few games go differently and the leaderboard can look very different.

Thanks for your attention!
