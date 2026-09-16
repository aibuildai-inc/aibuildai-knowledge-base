# 9th Place Solution

Competition: optiver-trading-at-the-close
Rank: #9
Source: https://www.kaggle.com/c/optiver-trading-at-the-close/discussion/486868

A big thanks to Optiver and Kaggle for hosting this competition. This competition has a really stable correlation between local cv and lb. 

Actually I entered this game a little late, about 30 days before its ends and I am not good at NN, so I only focus on Gradient Boosting tree models and its feature engineering. I noticed there are many top solutions using NN and it is really a good opportunity for me to learn NN.

### Model
- Xgboost with 3 different seeds and same 157 features
    - There is not much difference between Xgboost and Lightgbm in lb score. But GPU Xgboost trains faster than GPU Lightgbm.

### Feature Engineering
- Firstly, create some "basic features" based on raw features(i.e. add, subtract, multiply, divide from raw features). Also, create some median-scaled raw size features.

```python
size_col = ['imbalance_size','matched_size','bid_size','ask_size']
for _ in size_col:
    train[f"scale_{_}"] = train[_] / train.groupby(['stock_id'])[_].transform('median')
```
- Secondly, do further feature engineering/aggregation on raw features and "basic features"
    - imb1, imb2 features
    - market_urgency feateures I copied from public notebook
    - diff features on different time window
    - shift features on different time window
    - rolling_mean/std features on different time window
    - using history wap to calculate target of 6 second before. Then, do some rolling_mean 
    - some global date_id+seconds weighted features
    - MACD feateures
    - target rolliong_mean over stock_id + seconds_in_bucket

### Feature Selection
- Because we have limit on inference time and memory, it's essential to do some feature selection. I add features group by group and check whether the local cv improves. Each feature group usually have 10 - 30 features. If one groups make local cv improve, I add feature one by one insides this feature group and usually kept only 5-10 most effective features.
- I keep 157 features in my final model.

### Post-processing:
- Subtract weighted sum. From the definition of target, we can know weighted sum of target for all stocks should be zero.

```python
test_df['pred'] = lgb_predictions
test_df['w_pred'] = test_df['weight'] * test_df['pred']
test_df["post_num"] = test_df.groupby(["date_id","seconds_in_bucket"])['w_pred'].transform('sum') / test_df.groupby(["date_id","seconds_in_bucket"])['weight'].transform('sum')
test_df['pred'] = test_df['pred'] - test_df['post_num']
```

### Others:
- xgb mae objective
- xgb sample_weight 1.5 weight for latest 45 days data 
- Online training. I only retrain model twice. one is N day (N is the start date of private lb), the other is N+30 day.
- polars and `reduce_mem_usage function` helps a lot

### Codes
train: https://github.com/ChunhanLi/9th-kaggle-optiver-trading-close
inference: https://www.kaggle.com/code/hookman/9th-submission
