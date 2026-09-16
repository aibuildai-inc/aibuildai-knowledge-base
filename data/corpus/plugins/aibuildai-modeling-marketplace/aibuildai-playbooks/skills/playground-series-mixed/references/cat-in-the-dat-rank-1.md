# 1st place simple solution

Competition: cat-in-the-dat
Rank: #1
Source: https://www.kaggle.com/c/cat-in-the-dat/discussion/121356

I thought about whether to post my solution for a long time because it may not be better than other complicated solutions. 


But it is really simple and performs well. It’s more like a baseline model. I hope you can learn something from this. : )

What I did:
- Dropping bin_0
- Ordinal Encoding ord features
- One Hot Encoding other features
- Using logistic regression with ‘liblinear’ solver
- Tuning C by using [optuna](https://www.kaggle.com/cuijamm/simple-onehot-logisticregression-score-0-80801)

I did these things except optuna at the beginning of this game.
I tuned C the last day and trained model all at once instead of using Kfolds. It improved my score to 0.80850

I’m new to machine learning and this is my first competition. 
Thanks to Kaggle and everyone participates in this competition. I learned a lot from you guys, which is more important than the ranking. Thanks for your share!
