# 21st Place Solution: 5-Model Ensemble with Optuna Weight Optimization

Competition: march-machine-learning-mania-2026
Rank: #21
Source: https://www.kaggle.com/c/march-machine-learning-mania-2026/writeups/21st-place-solution-5-model-ensemble-with-optuna

I have been participating in this competition for years; this was my best submission by far. I wanted to share the general idea behind my approach this year. I leaned heavily on ensemble methods, hoping the variety of predictions would lead to a smoother overall submission. I'd like to pat myself on the back, but I recognize that this competition requires a lot of luck as well. My methods worked well for this year, next year they might not. 

In short, `"All models are wrong. Some models are useful." - George Box`

This year, my model was useful. Here is what I did. 

## Overview
My approach centered on a 5-model ensemble with gender-specific Optuna weight optimization. Models were trained separately for men and women, with walk-forward cross-validation used throughout to respect the temporal structure of the data.

## Feature Engineering
All features were computed as differentials (Team A minus Team B) to reduce multicollinearity and focus the model on relative team quality. Key features:

- MOV-adjusted ELO using a FiveThirtyEight-style margin-of-victory correction
- Pythagorean win percentage and last-10-game Pythagorean momentum
- Strength of schedule, consistency, and volatility
- Seed matchup win rates; historical, recent 5-year, and recent 10-year windows
- SeedDiff × Gender interaction term
- Gap average 

Training data was symmetrized, each game appears twice, once from each team's perspective. This doubles training data and eliminates perspective bias.

## Models
Five models were trained independently on the same feature set:
 
| Model | Notes |
| --- | --- |
| LightGBM | Best individual performer, gender-specific |
| XGBoost | Early stopping via callbacks (XGBoost 3.x)  |
| HistGradientBoosting | Tuned max_bins for differentiation from LGB |
| Logistic Regression  | StandardScaler applied, linear boundary adds diversity  |
| Neural Network  | Single wide layer 128-512 units, critical for women  |

</br>
Walk-forward validation used seasons 2019, 2021-2025 (excluding 2020 COVID) with weighted seasons emphasizing recent years.

## Ensemble
Optuna optimized model weights separately for men and women, using walk-forward Brier score as the objective. Final weights:

| Model | Men | Women |
| --- | --- | --- |
| LGB | 58.4% | 34.0% |
| LR | 21.9% | 8.5% |
| XGB | 18.4% | 10.7% |
| NN | 0.1% | 43.9% |
| HGB | 1.3% | 2.8% |

</br>
The NN weight divergence between men (0.1%) and women (43.9%) was the most interesting finding.

##Lessons Learned
*Don't chase upsets.* I wrote a separate post on this [here](https://www.kaggle.com/competitions/march-machine-learning-mania-2026/discussion/684446) with the full proof, but the short version: for any well-calibrated model, adjusting predictions post-hoc to account for upset probability is mathematically guaranteed to increase your expected Brier score. The excess cost is exactly (p̂ - p)² per game.

*HGB was redundant.* At 1-3% ensemble weight it contributed almost nothing. Next year I'd either force differentiation aggressively via binning strategy or drop it entirely.

*Gender-specific weights matter.* The ensemble weight divergence between men and women was large enough that a combined model would have left meaningful points on the table.

Also, from reading other write-ups, I have new things to try next year such as use of multiple seed values for each model, and the inclusion of Polymarket data.
