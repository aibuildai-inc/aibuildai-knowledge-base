# March Mania 2026: 10th Place Solution

Competition: march-machine-learning-mania-2026
Rank: #10
Source: https://www.kaggle.com/c/march-machine-learning-mania-2026/writeups/march-mania-2026-10th-place-solution

### Overview

The core idea was to reframe game outcome prediction as a **regression problem on point differential** rather than a binary win/loss classification. Predicting the margin of victory produces a richer training signal and naturally encodes the strength of a result, which a binary label discards. Final win probabilities were then recovered through a post-hoc calibration step.

The pipeline was applied jointly to both men's and women's NCAA basketball data, treating them as a single dataset with a gender indicator feature.

---

### Feature Engineering

All features were derived from regular season box score data and aggregated at the season level per team. The feature set covered several complementary angles:

**Season-averaged box score statistics:** Points, field goals (overall and three-point), free throws, rebounds (offensive and defensive), assists, turnovers, steals, blocks, and personal fouls, computed from both the team's perspective and the opponent's perspective. Box score totals were adjusted to normalize for overtime length.

**Possession-based efficiency metrics:** Possessions were estimated using the standard formula (FGA, offensive rebounds, turnovers, free throw attempts). From there, offensive rating, defensive rating, and net rating were derived per team.

**Shooting profile:** Effective field goal percentage (eFG%), true shooting percentage (TS%), three-point attempt rate, and free throw rate captured each team's offensive style and efficiency.

**Rebounding percentages:** Offensive and defensive rebound percentages, accounting for opponent rebounding rates.

**Venue splits:** Separate averages and volatility statistics computed for home, road, and recent games, with fallback logic for small sample sizes.

**Elo ratings:** A simple Elo system was trained over all regular season games each season, producing a dynamic strength rating per team. The feature used was the Elo difference between the two matchup sides.

**GLM team quality:** A Gaussian GLM was fit each season using point differential as the response, with team fixed effects for both sides. This produced a latent quality score per team, grounded in a principled statistical model. The feature used was the quality difference between the two teams.

**Seed difference:** The numeric tournament seed difference, directly encoding the committee's seeding judgment.

---

### Modeling

XGBoost was used as the base learner, trained to predict point differential directly. The training objective was regression, not classification.

**Validation scheme:** A leave-one-season-out cross-validation was applied over all seasons from 2003 onward. Each fold held out one complete tournament season, trained on all others, and produced out-of-fold (OOF) margin predictions for the held-out season. This respected the temporal structure of the data and provided unbiased OOF predictions spanning the full historical range.

**Hyperparameter and feature selection:** Optuna was used to jointly optimize XGBoost hyperparameters (learning rate, tree depth, subsampling ratios, regularization terms, number of trees, etc.) and perform feature selection, with the OOF Brier score as the objective.

**Multi-seed ensemble:** The full pipeline was run across six distinct random seeds. Each seed produced its own Optuna study, its own selected feature subset, and its own set of OOF-calibrated models. Final predictions were averaged across all seeds, providing diversity both in the feature subsets and in the stochastic aspects of the boosting procedure.

---

### Probability Calibration

Raw XGBoost outputs are point differential predictions, not probabilities. To convert them, a calibrator was fit on each seed's OOF margin predictions against the binary win indicator derived from those same margins. A symmetric clipping (trim) was applied to the raw margins before fitting, with the trim value itself treated as an Optuna hyperparameter.

At inference time, the same trim was applied to test predictions before passing them through the calibrator. Probabilities were clipped to the unit interval as a final safeguard. The ensemble prediction was the mean probability across all seeds and all season-specific models.
