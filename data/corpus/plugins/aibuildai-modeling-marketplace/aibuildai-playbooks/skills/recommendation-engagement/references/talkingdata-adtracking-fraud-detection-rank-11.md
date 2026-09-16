# 11th place features

Competition: talkingdata-adtracking-fraud-detection
Rank: #11
Source: https://www.kaggle.com/c/talkingdata-adtracking-fraud-detection/discussion/56250

Congrats winners!

I would like to share my features briefly before I forget it. 

### Features

I used BigQuery on the almost all feature engineering parts.  
 https://gist.github.com/tkm2261/1b3c3c37753e55ed2914577c0f96d222

It takes only about 20 minutes. very fast!

There may not be surprising features if you have seen the shared kernels. I tried to find good features by brute force way.

### Machine

CPU: 16 cores, MEM: 100GB on GCP

### Data

* training data: day 7 and day 8 (13,188,695,398 rows)
* validation data: day 9 (53,016,937 rows)
* test data: day 10 (18,790,469 rows)

I used validation data to determin the boosting round. Then, I used train+valid data for training.

### Training

Simply, I used LightGBM and ensembled with different seeds.

The single best model is 0.9823 on the public LB.

The parameter is here: 

    {'colsample_bytree': 0.6, 'learning_rate': 0.1, 'max_bin': 1023, 'max_depth': -1, 'metric': ['binary_logloss', 'auc'], 'min_child_weight': 30, 'min_split_gain': 0.0001, 'num_leaves': 127, 'objective': 'binary', 'reg_alpha': 0, 'scale_pos_weight': 1, 'seed': 1142, 'subsample': 0.9, 'subsample_freq': 1, 'verbose': -1}

source code is here: https://github.com/tkm2261/kaggle_talkingdata/blob/master/protos/train_lgb.py

I tried to earn my tuition for MS CS program starting next fall. But, there was no free lunch!
