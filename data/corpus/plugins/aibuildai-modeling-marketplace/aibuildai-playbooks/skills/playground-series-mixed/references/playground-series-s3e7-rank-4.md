# 4th Place Solution (Simple)

Competition: playground-series-s3e7
Rank: #4
Source: https://www.kaggle.com/c/playground-series-s3e7/discussion/390962

## Feature Engineering:
None

## CV:
Stratified KFold with 10 splits. I was happy with this competition because unlike the past few, this one had a relatively stable correlation between cv and lb. My final solution's cv score was 0.90667. 

## Models:
Lightgbm, XGBoost, CatBoost (of course). Each model was tuned by optuna to maximize the auc score for a validation set. 

## Ensemble: 
For each fold, the optimal weighted average was calculated between the predictions from each of the three models. To calculate the weights, I used scipy to minimize the negative auc score.

## Post-Processing:
I used the code suggested in this [discussion](https://www.kaggle.com/competitions/playground-series-s3e7/discussion/388992).

## What didn't work:
Feature engineering -- new features I added had little impact on the score
Pseudo Labeling -- I attempted to pseudo label the original test set, as well as the competitions test set but I didn't find success (improvement in cv, but sharp decline in lb. Probably overfitting). Soft labels also didn't work.

## Code:
https://www.kaggle.com/code/ryanbarretto/4th-place-solution?scriptVersionId=120547715
