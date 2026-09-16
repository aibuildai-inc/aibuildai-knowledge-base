# 8th solution. XGBoost and Dataset balancing

Competition: LANL-Earthquake-Prediction
Rank: #8
Source: https://www.kaggle.com/c/LANL-Earthquake-Prediction/discussion/94718#latest-546332

Many thanks organizers for such interesting competition.

And thanks authors for their very useful kernels:
https://www.kaggle.com/artgor/earthquakes-fe-more-features-and-samples
https://www.kaggle.com/gpreda/lanl-earthquake-eda-and-prediction
https://www.kaggle.com/allunia/shaking-earth
https://www.kaggle.com/vettejeep/masters-final-project-model-lb-1-392

I try different models: "time series models"- WaveNet, LSTM, ResNet-1d, FCN-1d.
"Features models" - XGBRegressor, DNN, SVR.
And XGBoost model was the best.
This is hyper parameters:
model = xgb.XGBRegressor(booster='dart',
                         tree_method='hist',
                         n_estimators=100000,
                         learning_rate=0.01,
                         max_depth=3,
                         subsample=0.9,
                         colsample_bytree=0.5,
                         reg_lambda=1,
                         gamma = 1)

And I noticed that the original training data set is unbalanced - there is few data with a time to failure more than 8 seconds.
Therefore, when creating my own dataset for thrain the model, I used data from 2, 7, 14 "long" quakes "more" than others, and used 4th quake for validation. 
This improved the prediction for longer times to failure.
