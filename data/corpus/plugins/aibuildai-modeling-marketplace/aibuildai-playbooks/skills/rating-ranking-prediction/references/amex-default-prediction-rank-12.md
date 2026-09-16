# 12th Place Gold (2/2) - LGBM + XGBoost + Catboost

Competition: amex-default-prediction
Rank: #12
Source: https://www.kaggle.com/c/amex-default-prediction/discussion/348058

Thanks to AMEX, Kaggle, and all participants for this great competition!

This post is the second part of our team's ensemble solution for 15th place. The first part written by my great teammate @joseantonioalatorre can be seen [here](https://www.kaggle.com/competitions/amex-default-prediction/discussion/347740).

TLDR: My part of the ensemble consists of LGBM, XGBoost, and Catboost models. In addition to the standard features, my most important features were model predictions for each individual statement and aggregations of only the last 3 statements of each customer. The best model was a dart-LGBM with early stopping.



# Features
I only used the integer dataset provided by @raddar. Here are some additional features that I didn't see in public notebooks:

- As each customer has several statements, I trained a model on the individual statements and created a dataset based on the aggregated predictions for each customer (mean, std, last...). (similar to the approach [here](https://www.kaggle.com/competitions/amex-default-prediction/discussion/347786))
- Aggregations of only on the last 3 statements for each customer
- Aggregations for the time between two statements
- 10 PCA dimensions
- Coefficient of variation (https://en.wikipedia.org/wiki/Coefficient_of_variation)

# Models
I used a diverse set of models, with the dart-LGBM performing best overall. For all models, I used 10-fold cross-validation and trained on binary cross-entropy. Overall my dataset consisted of about 2-3 thousand features. Hyperparameter-tuning or feature selection did not seem to help much. 
I should note that a huge part of my time went into finding ways to prevent memory errors from training with that many features.

## LGBM
I used dart with a custom early stopping callback on the validation metric. I trained several models with small changes in features and parameters. The parameters are mostly similar to the public kernels. I did however not transform the features to 16bit.

## XGBoost
This model was trained on the GPU in a similar style as the LGBM.

## Catboost
The Catboost was the hardest to get right, especially since no early stopping or callbacks were available for GPU training. This model's predictions have very little weight in the ensemble, however, they brought a significant boost to the final score.


### What I tried that did not work:
- Neural Networks
Couldn't get it working, and when I thought I did, the public score was abysmal.
- Autoencoder-Features
- Cluster aggregations
- Using a model to predict whether a statement is the last of a customer
- ....

I learned a lot in this competition, met new friends and got a very pleasant result. Huge success for me.
