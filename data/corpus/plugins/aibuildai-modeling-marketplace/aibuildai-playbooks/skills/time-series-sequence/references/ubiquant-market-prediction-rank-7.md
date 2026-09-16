# 7th place solution, single model, no supplemental data used

Competition: ubiquant-market-prediction
Rank: #7
Source: https://www.kaggle.com/c/ubiquant-market-prediction/discussion/338293

Hi,
A bit of self-introduction first. I currently work for a small hedge fund in Beijing. I came to this competition to test some of my ideas from work, and see if they apply to a different set of features. My solution is really a baseline model, with no reverse engineering, no ensemble, and even the supplemental training data provided were not used. 

I won't disclose my full solution but here are some facts about the model.

Model and hyperparameter:
The model is a single LGB model with hand-tuned parameters, no tunning packages were used. The 'extra_trees' pamameter is set to 'True'. This gives steady improvement when the number of trees goes large.

Feature engineer and selection:
This is kind of my secret. The model takes in 900+ features, which are selected from an even larger feature pool.
As a control, in the second submission I have a similar model but only used the 300 original features. That one scored 0.112 which is not even in the medal range.

Cross-validation：
Standard TimeSerieseSplit applies.

I guess I got lucky, but not in a way of a magical seed but most people are unfirmiliar with financial markets and thus overfitting on the public leaderboard. Thanks to everyone who participated in the competition.
