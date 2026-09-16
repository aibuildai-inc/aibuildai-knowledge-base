# #43 Explainable model almost without machine learning

Competition: playground-series-s3e6
Rank: #43
Source: https://www.kaggle.com/c/playground-series-s3e6/discussion/389155

Sometimes, proper methodology is useful in these competitions. When the public leaderboard was dominated by notebooks which skipped cross-validation and [trained on subsets of the available data](https://www.kaggle.com/competitions/playground-series-s3e6/discussion/384915), I decided to compare different cross-validation strategies. These experiments showed that samples with duplicated `squareMeters` should be treated differently from samples with unseen `squareMeters`:

A. When I cross-validated with `GroupKFold(groups=train.squareMeters)`, linear regression gave the best results, better than all tree-based models:

[horizontal bar chart]

B. When I cross-validated with `KFold`, I got the best results by predicting the mean price of all houses with identical  `squareMeters`, `made` and sometimes `cityCode` (this is the method I announced in [Quasi-duplicates in the data](https://www.kaggle.com/competitions/playground-series-s3e6/discussion/386116)). 

My [final model](https://www.kaggle.com/code/ambrosm/pss3e6-quickstart-without-machine-learning) includes the original dataset and uses only three features: `squareMeters`, `made` and `cityCode`. It predicts test prices in three phases:
1. If the training data contains houses with identical `squareMeters`, `made` and `cityCode`, predict the mean price of these houses.
2. If the training data contains houses with identical `squareMeters` and `made` (regardless of `cityCode`), predict the mean price of these houses.
3. Otherwise do a linear regression.
