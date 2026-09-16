# Two XGBoost models + rainfall fraction per group

Competition: playground-series-s5e3
Rank: #6
Source: https://www.kaggle.com/c/playground-series-s5e3/discussion/571216

I calculated two OOF with two XGB models, one for the training data and one for the original data. Then combined the predictions with .3 weight for the original data. I also added extra features by grouping by each column and calculating mean, sd, skew and rainfall fraction. What was most prominently picked up by the XGB was the rainfall fraction  (the sum of days in a group divided by the rainy days). I used the same inner fold outer fold structure as Chirs D. last month. That didnt result in a fantastic public leaderboard score but surprisingly did very well for the private LB score. 
```    model = XGBRegressor(
        objective="reg:logistic",
        max_depth=6,  
        colsample_bytree=0.9, 
        subsample=0.9,  
        n_estimators=10000,  
        learning_rate=0.1,  
        enable_categorical=False,
        early_stopping_rounds=100,
        verbosity=2,
        eval_metric=['auc'],
    )```
