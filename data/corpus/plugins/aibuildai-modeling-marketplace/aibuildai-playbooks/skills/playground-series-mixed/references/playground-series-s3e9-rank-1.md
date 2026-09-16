# #1 Solution: Cross-validation and diversity win

Competition: playground-series-s3e9
Rank: #1
Source: https://www.kaggle.com/c/playground-series-s3e9/discussion/394592

My final solution differs only slightly from what I published in my [EDA which makes sense](https://www.kaggle.com/code/ambrosm/pss3e9-eda-which-makes-sense). It is based on a few principles:
1. **Optimize the cv score and don't look at the public leaderboard!** The cv score is based on 5407 samples while the public leaderboard is based on only 721 samples. In a Kaggle competition, a 5407-sample average is a measurement - a 721-sample average is random variable. If you want to test the quality of a dice, you better throw it eight times rather than only once. This rule has a few consequences:
  - You don't need more than two submissions because you can't gain any information from your public leaderboard score. Ok, I used seven submissions because I was curious.
  - Don't copy code from high-scoring public notebooks unless the quality of that code shows up in a good cross-validation score. The best-scoring public notebooks are at the top of the list only because they overfit the public leaderboard. 
2. **Cross-validate correctly:** An ordinary `KFold` is enough for this competition. A `train_test_split` is not.
3. Implement a **diverse** ensemble!
  1. Everybody uses gradient boosting, but you need to find good hyperparameters. Optuna doesn't. Just run Optuna and then change the seed of the KFold. You'll see that the Optuna-found hyperparameters don't survive the change of seed. I optimized the LGBM hyperparameters manually.
  1. Using more than one gradient boosting implementation adds diversity to the ensemble. I used LightGBM and `GradientBoostingRegressor`.
  2. Random forests are simple to optimize: The most important hyperparameter is `min_samples_leaf`.
  3. Whereas the tree-based algorithms don't need much feature engineering, linear regression does. After the partial dependence plots of the EDA showed the nonlinearity of the game, I spent quite some time creating features. It is important to not just add features and hope for the best - in the same process you have to drop the features which don't improve the cross-validation score. 
4. Remember not to optimize your ensemble against the public leaderboard. Optimize the cv score.
5. `AgeInDays` has only a few different values and its relationship to the target is highly nonlinear and not monotone. I treated it as a categorical value and target-encoded it, replacing the feature values by the corresponding average target values. This target encoding even helped for some of the tree models.

[bar chart]

[Source code is here.](https://www.kaggle.com/code/ambrosm/pss3e9-winning-model)
