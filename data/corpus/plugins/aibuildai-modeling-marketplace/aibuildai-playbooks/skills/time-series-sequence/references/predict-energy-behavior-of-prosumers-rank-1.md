# 1st place solution

Competition: predict-energy-behavior-of-prosumers
Rank: #1
Source: https://www.kaggle.com/c/predict-energy-behavior-of-prosumers/discussion/472793

Thanks to Enefit and Kaggle for hosting this great competition. And thanks to the
great notebooks and discussions, I learned a lot. ~~I have a feeling this comp is going to be slightly more volatile 
than optiver one and wish I could end up in top place.~~ I am so happy to win my third solo win! 😃😀😀

# Overview
My final solution consists of 4 XGBoost models (for two targets and two is_consumption) and 2 GRU models (for two targets). These models share same 600 features. 
| model name |  validation set | test set w/o online training | test set w/ update one time | test set w/ update three times|
| --- | --- | --- | --- | --- |
| XGBoost | 41.5912(2 targets:42.2531/42.6607) | 55.6231 | 54.1343 | 54.1213 |
| GRU | 43.0815(2 targets:43.7182/45.1687) | 56.6651 | 54.9948 | 54.6960 |
| XGBoost+GRU|40.6790(2 targets:40.9613/41.5294)  | 53.6627 | 52.4813 | 52.3090 |

# Validation Strategy
My validation strategy is pretty simple, train on first 500 days and set the rest days as my holdout validation set which has good indication with leaderboard score. And I only checked the final overall score and didn't print single model's score.

# Features
I think my features are not special compared with other top teams. I simply merge all tables excluding gas_df with train_df and create lots of lagged features. I didn't spend much time to cutdown feature size since we have not much memory pressure in this competition. With careful feature design, I believe the final feature size can be reduced to around 100-200. Maybe generating different features and targets according to different conditions will achieve better model performance after reading other kagglers' solutions, but this is also very time consuming for me.

fw_new_feature from [this discussion](https://www.kaggle.com/competitions/predict-energy-behavior-of-prosumers/discussion/468654) help a bit(about 0.1~0.2), Thanks to @mohammadpakdaman0  

```
# client_df
client_df = client_df.drop(["date"]).sort(["data_block_id"], descending=False)
df = df.join(client_df.select(["county", "is_business", "product_type", "data_block_id","eic_count","installed_capacity"]), how="left", on=["county", "is_business", "product_type", "data_block_id"])
# electricity
electricity_df = electricity_df.drop(["origin_date"]).rename({"forecast_date":"datetime"}).with_columns([(pl.col("datetime").str.to_datetime()+pl.duration(days=1)).alias("datetime"), (pl.col("euros_per_mwh").abs()+0.1).alias("euros_per_mwh"),])
df = df.join(electricity_df[["data_block_id","datetime","euros_per_mwh"]], how="left", on=["data_block_id","datetime"])

# fw_df 
fw_df = fw_df.with_columns(
                (pl.col("origin_datetime").str.to_datetime()+pl.duration(hours="hours_ahead")).alias("datetime"),
                pl.col("latitude").cast(pl.datatypes.Float32).round(1),
                pl.col("longitude").cast(pl.datatypes.Float32).round(1),
                pl.col("data_block_id").cast(pl.datatypes.Int64),
            ).join(weather_station_to_county_mapping.drop(["county_name"]).with_columns(
                pl.col("latitude").cast(pl.datatypes.Float32).round(1),
                pl.col("longitude").cast(pl.datatypes.Float32).round(1),
            ), how="left",on=["longitude","latitude"]).drop(["longitude","latitude","origin_datetime"])
fw_df = fw_df.group_by(["county","datetime","data_block_id"]).agg([pl.mean(col).alias("fw_{}".format(col)) for col in forecast_weather_cols]).with_columns([pl.col("county").cast(pl.datatypes.Int64),    
                                                                    pl.col("data_block_id").cast(pl.datatypes.Int64),])
df = df.join(fw_df, how="left", on=["county","datetime","data_block_id"]).with_columns([(pl.col("installed_capacity")*pl.col("fw_surface_solar_radiation_downwards") / (pl.col("fw_temperature") + 273.15)).alias("fw_new_feature"),])

# hw_df 
hw_df = hw_df.with_columns(
            (pl.col("datetime").str.to_datetime()+pl.duration(days=1)).alias("datetime"),
                pl.col("latitude").cast(pl.datatypes.Float32).round(1),
                pl.col("longitude").cast(pl.datatypes.Float32).round(1),
                pl.col("data_block_id").cast(pl.datatypes.Int64),
            ).join(weather_station_to_county_mapping.drop(["county_name"]).with_columns(
                pl.col("latitude").cast(pl.datatypes.Float32).round(1),
                pl.col("longitude").cast(pl.datatypes.Float32).round(1),
            ), how="left",on=["longitude","latitude"]).drop(["longitude","latitude"])
hw_df = hw_df.group_by(["county","datetime","data_block_id"]).agg([pl.mean(col).alias("hw_{}".format(col)) for col in historical_weather_cols]).with_columns([                                                            pl.when(pl.col("datetime").dt.hour()>10).then(pl.col("datetime")+pl.duration(days=1)).otherwise(pl.col("datetime")).alias("datetime"),
                                                                pl.col("county").cast(pl.datatypes.Int64),    
                                                                pl.col("data_block_id").cast(pl.datatypes.Int64),])
df = df.join(hw_df, how="left", on=["county","datetime","data_block_id"])
```

# target
1. (target-target_shift2)/installed_capacity.
2. target/installed_capacity.
3. target-target_shift2
4. raw target

I only use 1&2 since 3&4 have no benefit to my final overall score. 

# model 
I didn't spend much time to tune my xgboost models, just simply train and predict.  And I find there is no team using NN models yet, which really surprises me.
My GRU models are also very simple.  input tensor's shape is (batch_size, 24 hours, 600 dense_feature_dim + 6*16 cat_feature_dim), followed by 2 layers GRU, output tensor's shape is (batch_size, 24 hours). Categorical features are ['county', 'product_type', 'hour', 'month', 'weekday', 'day']. NN models boost my final score by 0.9.


#online learning strategy

Although online learning didn't help much in public leaderboard, I think it is crucial in private leaderboard since we  
will have additional 8 months‘ data. So I just gave up chasing higher public leaderboard score by ensembling with more models and decided to retrain less models every month.

# What not worked for me 

1. solar features from this [notebook ](https://www.kaggle.com/code/karakasatarik/1st-place-solution-public-1-546-priv-1-488)
2. 1dcnn model and transformer model
3. multi-days input instead of singe day input when applying GRU models
4. split into more models given is_business equals 0/1. 

Thank you all! I will write more if necessary.
