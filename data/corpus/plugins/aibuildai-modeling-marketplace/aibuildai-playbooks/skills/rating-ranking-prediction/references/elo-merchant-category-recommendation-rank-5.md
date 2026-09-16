# #5 solution

Competition: elo-merchant-category-recommendation
Rank: #5
Source: https://www.kaggle.com/c/elo-merchant-category-recommendation/discussion/82314

Congrats to all winners and all who survived shakeup.

My solution consists of a few models:

1. Regression (full dataset)
2. Binary classification for outliers prediction (cards without repeated transactions in test period).  
     This model was used to split train and test datasets by threshold 0.015 for creation 2 other regression models (low and high probabilities). 
     I also used predictions from this model as a feature for high prob model (stacking with predictions from  full regression model)
3. Low prob model - regression with small concentration of outliers based on predicts of binary classification. For rare outliers I used decreased weights 0.4.
4. High prob model - regression with high concentration of outliers based on predictions of binary classification. The main features of this model were predictions from binary classification and full regression. Plus some other features. This model helped me to avoid post processing which was main source of shakeup. 
5. For submit I combine together predictions from low and high prob models and blend them with full regression.


Features

I created and checked a few thousands features. Finally, after selection, I used about one hundred features.

Some of the features were created by target encoding (target as is or binary target). 
Target encoding was two step process - first I calculated target means for features combinations and joined results with transactions and after that I aggregated to cards level. For target means and aggregation I used weights with decline for early month_lags.

I believe that train-test split was random and it was a good strategy to trust local cross validation instead of public leaderboard.
