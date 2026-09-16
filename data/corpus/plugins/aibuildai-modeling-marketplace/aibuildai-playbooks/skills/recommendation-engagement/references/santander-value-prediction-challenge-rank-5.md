# 5 place mini write-up

Competition: santander-value-prediction-challenge
Rank: #5
Source: https://www.kaggle.com/c/santander-value-prediction-challenge/discussion/63822

There was 87 groups of 40 features. A lot of feature engineereng (~900 features) and some feature selection (drop to ~500 features). There was a blend of 7 submits, but the most weighted submit (weight 0.85) was from Kazanova's StackNet.

There was 2 layers in this stacked model. 8 models (different regressors) at the first layer and a simple linear regression at the second.

Groups finding I do by myself and leak usage took from https://www.kaggle.com/c/santander-value-prediction-challenge/discussion/61472#363394
