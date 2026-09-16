# 17th Place Solution | Only 1 Catboost

Competition: playground-series-s3e17
Rank: #17
Source: https://www.kaggle.com/c/playground-series-s3e17/discussion/419648

|Local CV |Private    |Public    |
|---           |---           |---         |
|0.98125  | 0.98426 | 0.97764|

Here are my notes. The starred ones are probably the differentiators.
Dataset:
- *Use train data & original train: +0.002
- Only remove id and UID columns
- Remove duplicates from combined train dataset
- 3 new features. Seems to help public leaderboard (+0.004), but no difference on private (~ 0.000)
1) tmp['tratio'] = tmp['proc_temp'] / tmp['air_temp']
2) tmp['torque_mult_rspeed'] = tmp['torque'] * tmp['rspeed']
3) tmp['torque_mult_toolwear'] = tmp['torque'] * tmp['tool_wear']
- Used LabelEncoder to encode the categories
- Nothing done for dataset imbalance. Boosting algorithms should be able to handle this
- No feature scaling done. Boosting algorithm should be able to handle this
- Split into Train on 80%, validate 20%. Also, had a 5 fold stratified kfold option, but the 80/20 split seemed good enough, and is quicker to iterate on
- For final model, train on all data, and stop at the best iteration for Catboost

Modeling
- *Only 1 Catboost model. Picking Catboost for this dataset is important. Catboost generally seems to be slightly more accurate than Xgboost or LightGBM (although can be slower). It is not always true as there are competitions where LightGBM did better. Ensembling multiple Catboosts is probably better, but I did not try
- *Optuna + Wandb to sweep hyperparameters, and pick best one
- continuous columns: ['air_temp', 'proc_temp', 'rspeed', 'torque', 'tool_wear', 'tratio', 'torque_mult_rspeed', 'torque_mult_toolwear']
- categorical columns: ['PWF', 'pid', 'TWF', 'Type', 'OSF', 'RNF', 'HDF']
(pid = product id)
- *Use product id, and make sure to use the "cat_features" argument. (+0.013)
- *CPU accuracy > GPU accuracy. Catboost CPU and GPU have a few different settings. CPU is also deterministic, whereas GPU is not
- Used a smaller learning rate (0.025 vs 0.1 default)
- For Catboost, increasing border_count parameter may help (default CPU is 254)

Finally, LightGBMs should be able to get close to Catboost performance, but you need to add a "categorical_feature" to the LightGBM. I only discovered this in the last couple days.
