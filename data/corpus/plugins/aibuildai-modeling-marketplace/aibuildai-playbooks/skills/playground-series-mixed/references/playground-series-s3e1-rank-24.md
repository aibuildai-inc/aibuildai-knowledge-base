# 24th Place Solution 🏅 - My second ever Kaggle competition 🔥

Competition: playground-series-s3e1
Rank: #24
Source: https://www.kaggle.com/c/playground-series-s3e1/discussion/377993

**24th place!**🏅 I recently decided to get back into Kaggle competitions, and I’m very happy with my performance on this one. One year ago, I competed in the Sartorius Cell Instance Segmentation competition and it was one of the most rewarding things I’ve done in the past two years. The amount of information I learned in such a short period of time was astonishing. The same goes for this competition. Prior to this competition I hadn’t competed in any tabular competitions, and I again learned an incredible amount of information. To that end, I want to thank the Kaggle team for putting together these competitions and thank everyone who competed for making the competition enjoyable for noobs like myself. 

I would like to especially give a shout out to the following four notebooks and discussions that I continuously referenced throughout the competition, I couldn’t have done it without these (go give them an upvote):
- @phongnguyen1 - https://www.kaggle.com/code/phongnguyen1/distance-to-key-locations
- @dmitryuarov - https://www.kaggle.com/code/dmitryuarov/ps-s3e1-coordinates-key-to-victory
- @thedevastator - https://www.kaggle.com/competitions/playground-series-s3e1/discussion/376210
- @tilii7 - https://www.kaggle.com/competitions/playground-series-s3e1/discussion/376709

### My Solution:
- Quite frankly, I didn’t do anything groundbreaking. My solution consisted of an ensemble of XGBoost, LightGBM, and CatBoost using a 10 KFold split. I did some light hyperparameter optimization using Optuna for the XGBoost model, though not on the LightGBM or CatBoost model parameters.
- **Feature Engineering:**
    - Distance to any California city with over 500,000 population.
    - Encoding trick listed [here](https://www.kaggle.com/competitions/playground-series-s3e1/discussion/376210)
    - Distance to coastline features as listed in [this discussion](https://www.kaggle.com/competitions/playground-series-s3e1/discussion/376683)
    - PCA coordinates
    - Rotated coordinates (15, 30, 45)
    - Polar coordinates.
- **CV:** To compute my CV score, I used an 80/20 split of the training data and excluded the “original” dataset to get more accurate scores. 
- **Trusting local CV:** After playing around a bit with various models and feature engineering ideas, I decided to trust my CV score and determined that the top public leaderboard scores were either doing some crazy feature engineering or were slightly overfit. Trusting my CV was the right choice as I increased my position 24 spots in the private leaderboard :)

This competition will likely be the first of many for me this year, and hopefully you all will be seeing a lot more of me. I aim to play these tabular series until I land a top 3 position in one of them (I’m coming for you Kaggle merch).

You can find my solution notebook here: https://www.kaggle.com/code/ryanirl/ps-s03e01-main-notebook-xgb-lgbm-cb
