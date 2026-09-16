# 5. place solution

Competition: icr-identify-age-related-conditions
Rank: #5
Source: https://www.kaggle.com/c/icr-identify-age-related-conditions/discussion/430907

Knowing the inevitable shakeup, I submitted couple basic solutions months ago and forgot about the competition. I was surprised when my friends congratulated me in the morning.

[Here's the code](https://www.kaggle.com/code/celiker/icr-5-place-solution/notebook)

Main points:

1. Trained models for each Alpha, Beta, Gamma, Delta and stacked these probabilities to be used as features.
2. ####Created lgbm imputer models for every feature even if it has no missing values on train data.
3. Used RepeatedStratifiedKFold(n_splits=5, n_repeats=5) with a basic catboost model.

note: Removing imputers didn't effect the score, so the main strength is stacking greeks.

After this solution, I tried brute-force feature engineering and other modeling approaches, but they didn't help on public so I left it as final submission
