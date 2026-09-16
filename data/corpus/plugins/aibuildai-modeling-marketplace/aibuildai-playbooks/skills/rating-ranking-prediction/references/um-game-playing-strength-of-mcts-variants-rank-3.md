# 3rd place solution - two stage flip augmentation stacking with code

Competition: um-game-playing-strength-of-mcts-variants
Rank: #3
Source: https://www.kaggle.com/c/um-game-playing-strength-of-mcts-variants/discussion/549588

Thanks to UM organizers and kaggle team giving us such a interesting competition.This competition has no leak,stable cv-lb correlation,and lots of possibilities to improve, I really enjoyed exploring the accuracy boundaries of infinite possibilities.

# Overview
I think the key point is data augmentation and creating a stable cv lb pipeline(lb is more important), the private leaderboard has almost no shake show us public and private dataset should be the same distribution.



# Data Augmentation
filp the agent1-agent2 pair to agent2-agent1,change the AdvantageP1 to 1-AdvantageP1,change utility_agent1 to -utility_agent1,others keep same.

# Cross Validation
StratifiedGroupKFold by GameRulesetName, modify minor class of utility_agent1 to neighbour class.

# Feature Engingeering
I add some features from this great notebook https://www.kaggle.com/code/yunsuxiaozi/mcts-starter
And I also add tfidf-svd features of EnglishRules,this feature need to find the better parameter of max_features,I tried many times by lb score.

# Feature Selection
I like use null importance feature selection to drop those features will change too much if target shift

# Hyper-parameter tuning
I use optuna to search best hyper-parameters for lightgbm and catboost

# Models
I use two stage modeling,the first stage is using flip augmentation data as train data to predict original train data and test data,then generate oof predictions as the second stage's feature.

# Ensemble
My final submission is blending lightgbm and catboost with 3 seeds average ensemble.

# Post processing
The final prediction mean values are lower than original mean values, same with public lb, so I multiply coefficients(1.12) can improve lb score of 0.002.

# Not work
LudRules tfidf, bert embeddings,pseudo label with more agent1-agent2-games groups,neural network

# Notebook
https://www.kaggle.com/code/senkin13/um-final-sub-1
