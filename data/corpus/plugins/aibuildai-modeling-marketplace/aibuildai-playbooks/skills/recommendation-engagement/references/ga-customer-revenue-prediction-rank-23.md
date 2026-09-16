# 23th -place solution (Simple CatBoost)

Competition: ga-customer-revenue-prediction
Rank: #23
Source: https://www.kaggle.com/c/ga-customer-revenue-prediction/discussion/82263

Great thanks for my first serious comtetitions. I have joined it already at the second stage so I didnt feel disappointment for sudden changes rules :)
I used pretty standard aggregate features for my models, but i else calculated all features separately for all positive totalTransactionRevenue.
The absence of strong season in data allowed to use all presents data. I used each 5 month with target for 2 month pass one month. For example: x 2016/08/01 - 2017/01/01, y 2017/02/01 - 2017/04/01. For validation - just period from 2018/02/01 to 2018/08/01 with appropriate target.
I built standart CatBoost Regression with small number iteration and it was enough to prevent overfitting.
