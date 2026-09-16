# 8th Place Solution: Regression models for binary classification

Competition: playground-series-s3e10
Rank: #8
Source: https://www.kaggle.com/c/playground-series-s3e10/discussion/396425

Thank you all for another fun and exciting Playground Series episode! I will try to quickly go over the main points of my approach.

### Data

- I used `StandardScaler()` for feature scaling.
- I did not use the original dataset in my training data as I saw a consistent performance drop across many different models.

### Feature Engineering
At the beginning of the competition I spent a lot time experimenting with feature engineering. I thought there was definitely potential to create some interaction features between the mean, standard deviation, skewness, and kurtosis features. This included ideas like squaring the standard deviation to compute a variance feature and trying a bunch of different mathematical combinations. None of these experiments resulted in a significant increase in CV score. Then @siukeitin discovered that the `Skewness` and `EK` columns got swapped in the original data. I went back and tried the feature engineering ideas I had that included `Skewness` and `EK` again. Once again there was not a significant enough increase in CV score. The only effective idea I had was using 2nd degree PolynomialFeatures with LogisticRegression where I saw the CV go from around `0.038` --> around `0.032`.

### Models

Here is a list of (most of) the models I used in my ensemble with their CV scores:

- 'XGBRegressor': `0.03049`
- 'CatBoostClassifier': `0.03069`
- 'XGBClassifier': `0.03070`
- 'LGBMClassifier': `0.03073`
- 'LGBMRegressor': `0.03076`
- 'TFNN': `0.03177`
- 'MLPClassifier': `0.03198`
- 'HistGradientBoostingRegressor': `0.0323`
- 'GradientBoostingClassifier': `0.0323`
- 'LogisticRegression': `0.0324`
-  'RandomForestClassifier': `0.0325`
-  'ExtraTreesClassifier': `0.0325`

With TFNN being a Tensorflow neural network. So how does XGBRegressor relatively outperform the classifiers in a binary classification problem? The answer is confident predictions. XGBRegressor would predict values outside of the 0-1 range whereas the classifiers would not be as confident (never predicting exactly 0 or 1). I believe using a regression model worked well because of the strong separation between the target classes. Something interesting about the LogLoss metric is that it does not punish predictions for being outside of the 0-1 range (even though you could just clip the predictions to fall in this range).

I also used @paddykb's excellent [GAM model](https://www.kaggle.com/code/paddykb/ps-s3e10-gam-finger-on-the-pulsarrrrr) in my final ensemble as I theorized it would help to diversify the predictions.

Additionally, I tried using CatBoostRegressor but it ended up drastically underperforming vs the other regressors.

### Calibration

@sergiosaharovskiy suggested that probability calibration could help to prevent incorrect decisions based on the classifier's predictions, especially if those decisions are based on threshold values for the predicted probabilities. I experimented with calibration for all the models I was using. Here are the results:

Better with calibration:

- GradientBoosting
- LogisticRegression
- ExtraTrees
- RandomForest
- SVC
- KNearestNeighbors

Better without calibration:

- CatBoost
- XGBoost
- LGBM
- Neural networks

### Cross Validation

At the beginning of the competition I was using `StratifiedKFold` due to the large class imbalance. However, at some point I experimented with using just `KFold` and actually saw an improvement in CV score. I am still a little puzzled by this but I think this was because of the strong separation between the target classes.

That's pretty much all I have to say! Thank you to everyone once again and I hope to see you in the next episode!
