# 3rd Place Solution

Competition: playground-series-s3e8
Rank: #3
Source: https://www.kaggle.com/c/playground-series-s3e8/discussion/392824

Hey everyone! Thanks to all the competitors and organizers at Kaggle for this exciting competition!

Before sharing my solution, I want to mention that I have experienced several challenging shake-ups in the past 🙃, and as a result, my primary goal in this competition was to survive any shake-ups that might occur. Although we didn't experience any major shake-ups, my solution performed well thanks to my precious random seeds.

I must give credit to these amazing notebooks, so shoutout to their authors:

@sergiosaharovskiy's EDA and submission notebook: https://www.kaggle.com/code/sergiosaharovskiy/ps-s3e8-2023-eda-and-submission
@tetsutani's XGB+LGBM+CAT ensemble baseline notebook: https://www.kaggle.com/code/tetsutani/ps3e8-xgb-lgbm-cat-ensemble-baseline
@eamonntweedy's XGB+LGB+CB ensemble notebook: https://www.kaggle.com/code/eamonntweedy/playground-s3-e8-gemstones-xgb-lgb-cb-ensemble

My Strategy:

Basically, I tried to use as many features / feature combinations as possible. Then I prepared a custom script, and inside of this script a LGBM model runs n-1 columns iteratively, given n equals to length of X. The goal was to determine which features are useful and vice versa. If dropping a feature decreases my model's success, I dropped this feature for good.  While doing this, usage of many seeds was crucial therefore I've used different seeds, and took the average for scores. But a small improvement in RMSE can be tricky and it can cause damage to your model if std of folds is higher than default one. ( fully trained one with n columns )

So I made few different datasets based on my above approach. Like at first, I tried to decrease my RMSE, without even looking at std numbers. Then once I checked the std's, I prepared another 2 more datasets.  In my second attempt, I tried to decrease both the standard deviation and RMSEs simultaneously and at third attempt I tried to decrease only STD of scores. ( 0.015 was my tolerance threshold for RMSE's, so if dropping a feature cause +0.015 RMSE and -0.01 STD, I accepted it. ) Once I saw my std_avg decreased from 4.5'ish to 3.8'ish, I stopped the process.
(

At modeling part, I used optuna to tune parameters with a default value of n_iterations = 1000 / learning_rate = 0.1. After tuning the parameters, I increased n_iterations to 10000 and decreased learning rate to 0.01. ( ESR = 300) My final solution includes XGBoost + LightGBM + CatBoost predictions which are coming from these different datasets. ( like xgb1, xgb2, xgb3, lgb1, lgb2, lgb3 .. )

(btw i also used different seed combinations in parameter tuning&modelling too! At my latest sub, my predictions were coming from a train-predict loop which has 10 different seeds. That's why I named it as 'thats_the_code', I guess :P) 


