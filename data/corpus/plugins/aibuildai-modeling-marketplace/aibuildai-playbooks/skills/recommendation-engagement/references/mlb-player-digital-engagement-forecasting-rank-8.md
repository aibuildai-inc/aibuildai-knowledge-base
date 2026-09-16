# 8th place solution (Moro & taksai)

Competition: mlb-player-digital-engagement-forecasting
Rank: #8
Source: https://www.kaggle.com/c/mlb-player-digital-engagement-forecasting/discussion/271683

Thank you to the organizers for the fun competition and everyone who participated.
And thank you to my teammate ( taksai @tsaito21219 ).

I share our team's solution.

# summary:
- LSTM model
- valid: 2021-05, 2021-06, 2021-07
- ensemble LSTM:MLP:LightGBM=5:1:1

# 1. validation
- private-data(2021-08) is in-season, so we use in-season data as validation, and select the month near 2021-08.
- valid: 2021-05, 2021-06, 2021-07
- prepared some patterns as training-data while shifting the period. 
- trained 10-fold for each model

[[img1-validation.png]](https://postimg.cc/nCds5N1C)

# 2. preprocess data
- features: total number of 232
    - mean/median/std/min/max of target per playerId last month
    - join each table by key(date/playerId/teamId), use the almost feature of each table
    - use all tables except events.csv
    - not use target-lag-feature
- didn't use target-lag-feature, because it's risky. However, if spliting the model according to forecast date and use only fixed values, we may have improved the score.

# 3. model
- LSTM (keras)
    - input: features in forecast-day + past-days(to 5 days ago)
    - output: multi-output (target1, target2, target3, target4)
    - model: Input(6days) > TimeDistributed(Dense) > LSTM > LSTM > Dense(256>128>64>4)
- other model:
    - MLP (input: features in only forecast-day)
    - GBDT (input: features in only forecast-day)

[[img1-lstm.png]](https://postimg.cc/dDwL0JHh)

# 4. ensemble
- LSTM:MLP:GBDT = 5:1:1
- score in public LB (evaluation data: 2021-05)
    - LSTM: LB=1.28
    - MLP : LB=1.32
    - GBDT: LB=1.36
    - ensemble: LB=1.26

# 5. measures to avoid submission-errors
- add a lot of exception handling so that it works even if the table or data is missing.
- use same function in both training and predicting
- create dummy data of 2021-08 and 2021-09, and confirmed to work using dummy data on local PC and Kaggle'notebook.
- use API Emulator ( @nyanpn ). Thank you !
[API Emulator for debugging your code locally](https://www.kaggle.com/nyanpn/api-emulator-for-debugging-your-code-locally)

Thank you to my teammate. Let's do it together again!
