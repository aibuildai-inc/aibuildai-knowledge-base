# 38th place - 41 Models, Custom Features, and Hill Climbing

Competition: playground-series-s5e5
Rank: #38
Source: https://www.kaggle.com/c/playground-series-s5e5/discussion/582892

Hey!

This was my first proper attempt at a Kaggle competition and I'm super grateful for the supportive community we have here. I'm sharing my solution and learnings here in the hope that it helps others, just like many of your posts helped me!

---

## Approach Overview

I explored a variety of models (like catboost, xgboost, random forest, extra trees, multi layer perceptron, ridge regression, bayesian regression, etc.) and experimented with different features like:
- Groupby aggregates features
- KMeans cluster assignments
- Time-series inspired features
- Multiscale binning
- Rank-based transformations
- Mathematical transformations
- PCA components added as features
- Domain specific features etc.

Alongside exploring different sets of features, I also tried out different kinds of ensembles like simple weighted averages of CatBoost and XGBoost models, Ridge regression ensembles etc., adding in some creativity to the process as well. Eventually, I implemented a **hill climbing ensemble**  and that turned out to be my most effective strategy.

---

## Final Submission

After generating OOF predictions from **41 different models** with different sets of features as written before, I passed them to the hill climbing algorithm, and it selected 16 models for the final prediction:
- 9 CatBoost models
- 3 Extra Trees models
- 2 XGBoost models
- 1 MLP model
- 1 Ridge model

This gave me my best final CV score.

---

## Learnings & Improvements

- I wasn’t able to properly tune **CatBoost** and **XGBoost** models due to some kernel issues. Fine-tuning these models and using **positive-only ensemble weights** could have boosted my final score I guess, based on some discussions I have looked into.

- In hindsight, I should have trusted my CV score more than the public leaderboard. I have a few submissions that would've likely ranked between **20-25** but didn’t pick them because of leaderboard shakeups.

---

A huge thanks to @cdeotte for patiently answering my questions and for sharing so many valuable insights, they helped a lot during this competition!

As a complete beginner, I'm very happy to have participated. This community is fantastic and every question I posted was answered kindly and helpfully. I learned a ton through this competition and the discussions.

Looking forward to participating in more!
