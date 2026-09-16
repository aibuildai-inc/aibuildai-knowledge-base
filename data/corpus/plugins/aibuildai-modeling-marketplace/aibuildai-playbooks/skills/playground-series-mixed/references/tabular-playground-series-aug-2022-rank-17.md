# 17th Place Solution

Competition: tabular-playground-series-aug-2022
Rank: #17
Source: https://www.kaggle.com/c/tabular-playground-series-aug-2022/discussion/349541

I'm happy and surprised to have placed this well, me and my teammate were busy most of the month and didn't spend much time experimenting or trying to optimize our solutions. I imagine this saved us from overfitting to the public leaderboard :p

I'll share the basics of our solution here for posterity since I don't think our code is particularly presentable for sharing.

#### 1. Custom validation splits 

Split our training data such that we train on three product codes and validate on two. For the final prediction, we averaged our predictions over each split. Courtesy of [this notebook](purist1024) by @purist1024 

#### 2. Feature Engineering

We created four new features: 

1. Missing value indicator for m3
2. Missing value indicator for m5
3. Area feature using the product of attribute2 and attribute3.
4. Average of m3 through m17

#### 3. Imputation

Imputation scheme from [this notebook](https://www.kaggle.com/code/pourchot/hunting-for-missing-values) by @pourchot. We got similar results using the KNN as in the original notebook and using the median over each product code. Finally, we used the [RobustScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.RobustScaler.html) to scale the data.

#### 4. Feature Selection

Other than the 4 new features created previously, we *only* used the following features for training our models, all of the other features were discarded.

1. loading (we used the log of the original loading)
2. attribute_0 (encoded using the WOE encoder)
3. measurements 0, 1, 2, and 17

#### 5. Model Choice

[We tested a bunch of models](https://www.kaggle.com/code/rsizem2/tps-08-22-comparing-models) using mostly default settings and settled on the following:

1. Logistic Regression
2. [XGBoost with Linear Boosting](https://xgboost.readthedocs.io/en/stable/parameter.html#parameters-for-linear-booster-booster-gblinear)

Finally, we did a minimal hyperparameter search to optimize the regularization parameters. Our best (but unsubmitted) solution was using XGBoost. I don't think it is particularly well known that XGBoost has boosted linear models as an option. Any time there is a competition where logistic regression does well, I think it's worth trying.

As you might notice, almost all of these ideas were borrowed from things posted in the discussion forums or published in public notebooks. So thanks to all of the clever and generous people for sharing their notebooks and posting their ideas in the discussion forums.
