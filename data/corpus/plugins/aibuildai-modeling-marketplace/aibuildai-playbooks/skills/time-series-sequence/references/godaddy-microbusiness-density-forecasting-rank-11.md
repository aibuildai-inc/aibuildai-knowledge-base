# 11th Place Solution - Simplicity (Luck) Is All You Need

Competition: godaddy-microbusiness-density-forecasting
Rank: #11
Source: https://www.kaggle.com/c/godaddy-microbusiness-density-forecasting/discussion/417803

First, my thanks to Kagglers for very informative and helpful discussions and notebooks. Also, thank you to Kaggle and GoDaddy for hosting the competition.

My code is available at [github](https://github.com/ywugwu/GoDaddy-MDF-Competition), which is based on [VADIM KAMAEV's](https://www.kaggle.com/code/vadimkamaev/microbusiness-density) public notebook. 

In this competition, my idea is to keep improving my CV and PB score without letting my model grow too complex.

# Validation (CV)
I used the last 5 month data for validation. The model with the best validation score is used as one of my final submissions.

# Model
Same with the public notebook, I used a Catboost Regressor to stack a LightGBM, an XGBoost, and a CatBoost model. Tuning a lot hyperparameters.

# Feature Engineering
I didn't do too much novel feature engineering, only used techniques from public notebooks. 

But I change the hyperparameters a lot.

# Blending Submissions
I used different hyperparameters of the model and different features (i.e., with different diff, max, mean, etc. settings) to conduct a large number of submissions. I select the best 4 (or maybe 3) submissions and use a weighted average to create a blended submission. The best (according to the public leaderboard) blended result is selected as one of my final submission. In the end, the private leaderboard score of blended submission is slightly better than the best CV model.
