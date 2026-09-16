# 10th Place Solution

Competition: playground-series-s4e6
Rank: #10
Source: https://www.kaggle.com/c/playground-series-s4e6/discussion/515980

Firstly, I want to express my gratitude to the Kaggle team for organizing this competition.

Ironically,
Our Final Submission is same submission from AutoML GrandPrix Solution, which was `Bagged Tunned LGB + Bagged Tunned XGB(Voting)`

which scored` 0.83699` on Public LB and `0.83905` on Private LB.

`I applied Bagging with n_estimators=100 and predictions were obtained with full fit`
something like this,

```python

estimators = [
    ('lgb',BaggingClassifier(estimator=lgb,n_estimators=100)),
   ('xgb',BaggingClassifier(estimator=xgb,n_estimators=100)),
]

VotingClassifier(estimators=estimators,voting='soft')

```

[Detailed Solution ](https://www.kaggle.com/competitions/playground-series-s4e6/discussion/511712)

I wish they considered the private leaderboard scores for the AutoML GrandPrix 🥲.
