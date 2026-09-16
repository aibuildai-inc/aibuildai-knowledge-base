# 33rd Place Solution

Competition: um-game-playing-strength-of-mcts-variants
Rank: #32
Source: https://www.kaggle.com/c/um-game-playing-strength-of-mcts-variants/discussion/549889

I would like to thank the UM organizers and the Kaggle team for hosting such an interesting competition. With a bit of luck, I was able to achieve my best placement ever, and I’m absolutely thrilled!

---

## Overview

The result is essentially an ensemble of two CatBoost models.  
The primary differences between the two models lie in the features and hyperparameters.

---

## Feature Engineering

- **Features extracted from LudRules**  
- **Arithmetic operations** between features with high feature importance  
- **Removal of features** with a feature importance score of 0  
- The features used and those removed were determined based on the Public LB score. I believe this decision contributed significantly to the results.  
- Both Model 1 and Model 2 used mostly similar features, but for Model 2, **TF-IDF features** and features generated through **one-hot encoding** were excluded.

---

## Validation

- For both Model 1 and Model 2, **GroupKFold** was used with `'GameRulesetName'` as the group.  
- The number of folds:  
  - Model 1: N=5  
  - Model 2: N=10  

---

## Training Models

- **Model 1**: CatBoost with GPU support  
  - Used `'RMSE'` as the `eval_metric`  
  - Applied `'R2'` as a `custom_metric`  

- **Model 2**: CatBoost with GPU support  
  - Applied multiple `custom_metric` values: `['MAE', 'Tweedie:variance_power=1.5', 'Huber:delta=1.0', 'R2']`  

Since I lack experience and knowledge, the selection of these metrics was based on their impact on the Public LB. I also believe the results were relatively well-aligned with CV.

---

## Ensemble and Post-processing

- The ensemble was calculated as:  
  `Model 1 * 0.3825 + Model 2 * 0.7225`  

- Based on public code, post-processing to limit values above 1.1 was effective, so I applied it.

---

Thank you.
