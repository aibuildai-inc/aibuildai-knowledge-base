# 4rd Place Solution for the "ICR - Identifying Age-Related Conditions"

Competition: icr-identify-age-related-conditions
Rank: #4
Source: https://www.kaggle.com/c/icr-identify-age-related-conditions/discussion/431173

I'm pleasantly surprised to see such a strong perturbation on the leaderboard, resulting in me going to the very top of the leaderboard. No doubt, when I sent my 3rd attempt 2 months ago (0.16 on public), I certainly did not suspect that it would end up in 4th place on private (0.34). Since then, I've seen the score decrease rapidly on public and I've realized that all these solutions are heavily overfitted, which is something I've tried to avoid in every solution I've made. 

Now for the key features of my solution.

1) Recursive filling of gaps in features using regression on CatBoostRegressor (default hyperparameters),
2) greeks['Epsilon'] Unknown were filled with greeks['Epsilon'].min()
3) row_id - row number in train and in test when sorting by Epsilon
4) Creation features with CatBoostClassifier training for each value in 'Alpha', 'Beta', 'Gamma', 'Delta' - probabilities for corresponding values of these categories similarly https://www.kaggle.com/competitions/icr-identify-age-related-conditions/discussion/430907. To avoid overtraining for prediction on test, I used 5 - fold cross validation and then simple averaging,
5) Final model - CatBoostClassifier without any tuning of hyperparameters and feature elimination

In subsequent attempts I tried to expand the feature space by inventing various new features, complications of filling in the gaps, for example, I tried to predict epsilon and row_id, but I did not get any improvement in the results on cross validation, moreover, the results became more unstable, so I realized that these complications only lead to overfitting and I stopped these attempts.

My solution is https://www.kaggle.com/code/andrejvetrov/third?scriptVersionId=131958512
