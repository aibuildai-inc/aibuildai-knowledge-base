# 5th place solution, single NN model

Competition: ubiquant-market-prediction
Rank: #5
Source: https://www.kaggle.com/c/ubiquant-market-prediction/discussion/338400

Thanks to Ubiquant and everyone involved in the competition, especially to all the kaggers that shared their knowledge in some  amazin notebooks.
The idea for my notebook was made it as simple and stable as possible. 

Rank		Score			Update
1344		0.1481			publicleaderboard
16			0.0865			publicleaderboard_update1
24			0.1141			publicleaderboard_update2
11			0.1304			publicleaderboard_update3
13			0.1239			publicleaderboard_update4
5			0.1198			publicleaderboard_update_final



1. Training Data: train.csv (with time_id >599) + supplemental_train.csv 
2. Target log transformation and removed 127 Target outliers rows.
3. No Feature Engineering just transform features with sklearn QuantileTransformer.
3. Simple NN model with four dense layers(optimizer=Adam, loss='mse', metrics=[rmse,wcorr])
4. Custom Cross Validation for Training with 20 folds and 10 purge time_id.
