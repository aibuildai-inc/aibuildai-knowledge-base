# 3rd place solution

Competition: playground-series-s6e2
Rank: #3
Source: https://www.kaggle.com/c/playground-series-s6e2/writeups/3rd-place-solution

First, a huge thanks to all the competitors! Special thanks to @cdeotte for supplying the original dataset when it went missing—it felt exactly like when Gandalf arrived at the dawn of the fifth day! :). I also want to thank @include4eto for the XGB with pseudo-labels approach and @omidbaghchehsaraei for the RealMLP baseline.

My strategy focused on building models with different feature sets and objectives, checking the correlations of the predictions, and strictly trusting the CV score. 

**Final CV Score: 0.955803**

# Models

My final ensemble was a blend of Gradient Boosting, Neural Networks, and Linear Models. I used two different feature sets for all the models and blended the predictions in the final. Also, I noticed that lower depths for GBDT algorithms (2, 3) worked better on this data.

- Version 1: All features treated as categorical.
- Version 2: A "Hybrid" set where only features with **< 10** unique values were categorical, and the rest were treated as continuous.

1. RealMLP & TabM: RealMLP was my strongest performer with a CV score of 0.95576. I used **n_cv=2 and the 1-auc_ovr metric**, [inspired from this discussion](https://www.kaggle.com/competitions/playground-series-s6e2/discussion/674394).

2. CatBoost Variants: I utilized both Plain and Ordered boosting types. I found that Ordered boosting was particularly effective as the 2nd best single-model strategy with 0.0.95575 CV score.

3. XGBoost Variants: Standard XGBoost (GBDT) with low depth (depth=3), XGBoost with Pseudo-labeling, and XGBoost with Logistic Regression Residuals.

4. HistGradientBoosting & LGBM, Logistic Regression: Included for additional architectural diversity. Helped improve the ensemble CV score.

# Feature Engineering & The Original Data

The only feature engineering technique that worked for me was including the original dataset. I used it primarily for Target Encoding, calculating the mean of the target from the original data for each categorical feature and merging it into the competition folds. 

# Ensemble

I converted all OOF predictions to ranks using scipy.stats.rankdata and used a GPU-accelerated Hill Climbing algorithm.

I added my solution to my GitHub. It's not completed yet, but I will try to update it soon. [Solution link](https://github.com/mert-byrktr/KAGGLE-PS-S6E2-PREDICTING-HEART-DISEASE)
