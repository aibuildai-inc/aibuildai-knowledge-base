# While waiting for the final results.....What solutions/approaches were used

Competition: mlb-player-digital-engagement-forecasting
Rank: #4
Source: https://www.kaggle.com/c/mlb-player-digital-engagement-forecasting/discussion/256559#1408742

I spent majority of my time building efficient feature generation pipeline (Maybe little too much! I hope nothing fails on test run). The end output of all that software engineering is that **I am able to generate ~600 features on train data in 5 minutes** 😁.  Which essentially means, I can run training from scratch on new data updated till July 31st.

Strategy for 2 submissions were as follows:

Robust model: An ensemble of 6 LGB model (trained on different time periods and different feature sets) and 8 NN models. The validation score on different time frames is comparable for all models, (every time frame a different model is best)

Overfit to most recent data: I selected a notebook that will train 2 LGB models on all data till July 31st (till July 17th it took 5 hours to run so I might just squeeze in 6 hours of limit). To make results stable, added results of robust model with 50% weight.

Didn't spend much time on manual feature engineering, just threw kitchen sink at my models.

I plan to write detailed post on **efficient feature engineering pipeline for time x user** kind of datasets
