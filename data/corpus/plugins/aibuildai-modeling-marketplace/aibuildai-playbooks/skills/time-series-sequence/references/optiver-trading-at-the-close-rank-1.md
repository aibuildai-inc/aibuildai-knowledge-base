# 1st place solution

Competition: optiver-trading-at-the-close
Rank: #1
Source: https://www.kaggle.com/c/optiver-trading-at-the-close/discussion/487446

Thanks to Optiver and Kaggle for hosting this great financial competition. And thanks to the
great notebooks and discussions, I learned a lot. I am so happy to win my second solo win! 😃😀😀

# Overview
My final model(CV/Private LB of 5.8117/5.4030) was a combination of CatBoost (5.8240/5.4165), GRU (5.8481/5.4259), and Transformer (5.8619/5.4296), with respective weights of 0.5, 0.3, 0.2 searched from validation set. And these models share same 300 features.

Besides, online learning(OL) and post-processing(PP) also play an important role in my final submission.

| model name |  validation set w/o PP |  validation set w/ PP | test set w/o OL w/ PP | test set w/ OL one time w/ PP | test set w/ OL five times w/ PP |
| --- | --- | --- | --- | --- | --- |
| CatBoost | 5.8287 | 5.8240 | 5.4523 | 5.4291 | 5.4165 |
| GRU | 5.8519 | 5.8481 | 5.4690 | 5.4368 | 5.4259 |
| Transformer | 5.8614 | 5.8619 | 5.4678 | 5.4493 | 5.4296 |
| GRU + Transformer | 5.8233 | 5.8220 | 5.4550 | 5.4252 | 5.4109 |
| CatBoost + GRU + Transformer | 5.8142 | 5.8117 | 5.4438 | 5.4157 | 5.4030*(overtime)|

# Validation Strategy
My validation strategy is pretty simple, train on first 400 days and choose last 81 days as my holdout validation set. The CV score aligns with leaderboard score very well which makes me believe that this competition wouldn't shake too much. So I just focus on improving CV in most of time.

# Magic Features
My models have 300 features in the end. Most of these are commonly used, such like raw price, mid price, imbalance features, rolling features and historical target features.
I will introduce some features really helpful and other teams didn't share yet.
1 agg features based on seconds_in_bucket_group
```
pl.when(pl.col('seconds_in_bucket') < 300).then(0).when(pl.col('seconds_in_bucket') < 480).then(1).otherwise(2).cast(pl.Float32).alias('seconds_in_bucket_group'),
```
```
 *[(pl.col(col).first() / pl.col(col)).over(['date_id', 'seconds_in_bucket_group', 'stock_id']).cast(pl.Float32).alias('{}_group_first_ratio'.format(col)) for col in base_features],
 *[(pl.col(col).rolling_mean(100, min_periods=1) / pl.col(col)).over(['date_id', 'seconds_in_bucket_group', 'stock_id']).cast(pl.Float32).alias('{}_group_expanding_mean{}'.format(col, 100)) for col in base_features]

```
2 rank features grouped by seconds_in_bucket
```
 *[(pl.col(col).mean() / pl.col(col)).over(['date_id', 'seconds_in_bucket']).cast(pl.Float32).alias('{}_seconds_in_bucket_group_mean_ratio'.format(col)) for col in base_features],
 *[(pl.col(col).rank(descending=True,method='ordinal') / pl.col(col).count()).over(['date_id', 'seconds_in_bucket']).cast(pl.Float32).alias('{}_seconds_in_bucket_group_rank'.format(col)) for col in base_features],
```

# Feature Selection
Feature selection is important because we have to avoid memory error issue and run as many rounds of online training as possible. 
I just choose top 300 features by CatBoost model's feature importance.

# Model 
1. Nothing to say about CatBoost as usual, just simply train and predict.
2. GRU input tensor's shape is (batch_size, 55 time steps,  dense_feature_dim), followed by 4 layers GRU, output tensor's shape is (batch_size, 55 time steps).
3. Transformer input tensor's shape is (batch_size, 200 stocks,  dense_feature_dim), followed by 4 layers transformer encoder layers, output tensor's shape is (batch_size, 200 stocks). A small trick that turns output into zero mean is helpful.
```
out = out - out.mean(1, keepdim=True)
```

4 sample weight

#Online Learning Strategy
I retrain my model every 12 days, 5 times in total. 
I think most teams can only use up to 200 features when training GBDT if online training strategy is adopted. Because it requires double memory consumption when concat historical data with online data. 
The data loading trick can greatly increase this. For achieving this,  you should save training data one file per day and also loading day by day.

*data loading trick*
```
def load_numpy_data(meta_data, features):
    res = np.empty((len(meta_data), len(features)), dtype=np.float32)
    all_date_id = sorted(meta_data['date_id'].unique())
    data_index = 0
    for date_id in tqdm(all_date_id):
        tmp = h5py.File( '/path/to/{}.h5'.format(date_id), 'r')
        tmp = np.array(tmp['data']['features'], dtype=np.float32)
        res[data_index:data_index+len(tmp),:] = tmp
        data_index += len(tmp)
    return res
```
Actually, my best submission is overtime at last update. I just skip online training if total inference time meets certain value.
So there are 4 online training updates in total. I estimate that the best score would be around 5.400 if not overtime.
Anyway, I am really lucky! 

#Post Processing
Subtract weighted-mean is better than average-mean since metric already told.
```
test_df['stock_weights'] = test_df['stock_id'].map(stock_weights)
test_df['target'] = test_df['target'] - (test_df['target'] * test_df['stock_weights']).sum() / test_df['stock_weights'].sum()
```


# What not worked for me 
1. ensemble with 1dCNN or MLP. 
2. multi-days input instead of singe day input when applying GRU models
3. larger transformer, e.g. deberta 
4. predict target bucket mean by GBDT

Thank you all!
