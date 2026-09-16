# 1st place solution

Competition: favorita-grocery-sales-forecasting
Rank: #1
Source: https://www.kaggle.com/c/favorita-grocery-sales-forecasting/discussion/47582

Congrats to all winner teams and new grandmaster sjv.
Thanks to kaggle for hosting and Favorita for sponsoring this great competition.
Special thanks to @sjv, @senkin13, @tunguz, @ceshine, we build our models based on your kernels.

- https://github.com/sjvasquez/web-traffic-forecasting/blob/master/cnn.py
- https://www.kaggle.com/senkin13/lstm-starter/code
- https://www.kaggle.com/tunguz/lgbm-one-step-ahead-lb-0-513 
- https://www.kaggle.com/ceshine/lgbm-starter

Like the Rossmann competiton, the private leaderboard shaked up again this time. I think luck is on our side finally.   

------
## Sample Selection

we used only 2017 data to extract features and construct samples.

train data：20170531 - 20170719 or 20170614 - 20170719， different models are trained with different data set.

validition: 20170726 - 20170810

In fact, we tried to use more data but failed. The gap between public and private leadboard is not very stable. If we train a single model for data of 16 days, the gap will be smaller(0.002-0.003).

## Preprocessing

We just filled missing or negtive promotion and target values with 0.

## Feature Engineering

1. basic features
    - category features: store, item, famlily, class, cluster...
    - promotion
    - dayofweek(only for model 3)
2. statitical features: we use some methods to stat some targets for different keys in different time windows
    - time windows
        - nearest days: [1,3,5,7,14,30,60,140]
        - equal time windows: [1] * 16, [7] * 20...
    - key：store x item, item, store x class
    - target: promotion, unit_sales, zeros
    - method
        * mean, median, max, min, std
        * days since last appearance
        * difference of mean value between adjacent time windows(only for equal time windows)
3. useless features
    - holidays
    - other keys such as: cluster x item, store x family...
    
## Single Model

-  model_1 :  0.506 / 0.511 ,  16 lgb models trained for each day [source code](https://www.kaggle.com/shixw125/1st-place-lgb-model-public-0-506-private-0-511) 
- model_2 :  0.507 / 0.513 ,  16 nn models trained for each day [source code](https://www.kaggle.com/shixw125/1st-place-nn-model-public-0-507-private-0-513)
- model_3 :  0.512  / 0.515，1 lgb model for 16 days with almost same features as model_1
- model_4 :  0.517  / 0.519，1 nn model based on @sjv's code

## Ensemble

Stacking doesn't work well this time, our best model is linear blend of 4 single models.

final submission = 0.42*model_1 + 0.28 * model_2 + 0.18 * model_3 + 0.12 * model_4 

public = 0.504 , private = 0.509
