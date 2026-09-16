# 18th Place Gold

Competition: amex-default-prediction
Rank: #18
Source: https://www.kaggle.com/c/amex-default-prediction/discussion/349250

## Overview
This competition got me my first (solo) gold medal, so I am sure you can imagine how happy I am and how much I enjoyed it. Big thanks to the organizers and the kaggle team!

My work was also based on other people's great effort, so big thanks and shoutout to @cdeotte @raddar @ragnar123 and @thedevastator I will mention all their contributions that I used further.

I built a 3 stage model - 39 stage one base models, 2 stage two ensemble models and stage 3 is a simple average of the 2 stage two models.


## First Stage - 39 base models

I used same CV strategy for every model `fold = argsort(customer_id)%5`, for every model I generated out-of-fold predictions and test predictions averaged across folds.

### 1. lightgbm with [dataset](https://www.kaggle.com/datasets/raddar/amex-data-integer-dtypes-parquet-format) from @raddar 
- computed simple features, fractions of "first / last" feature value and "mean / last" value when features are nonzero in train and test, else difference of these combinations
- simple hand tuning of hyper-parameters
- 13 models altogether

### 2. [lightgbm](https://www.kaggle.com/code/ragnar123/amex-lgbm-dart-cv-0-7977) from @ragnar123 
- 8 models altogether
- 1 original model
- 5 models with tuned learning rates
- 2 scaled models: original_num_trees * N, original_learning_rate/N for N in [2,4]

### 3. [lightgbm](https://www.kaggle.com/code/thedevastator/amex-bruteforce-feature-engineering) from @thedevastator 
- 3 models altogether 
- 1 original model
- 2 scaled models: original_num_trees * N, original_learning_rate/N for N in [2,4]


### 4. my custom CNN implementation with custom dataset
- 13 models altogether
- different architectures (number of filters and convolution layers)


### 5. [transformer](https://www.kaggle.com/code/cdeotte/tensorflow-transformer-0-790) from @cdeotte 
- only 1 original model

### 6. gaussian naive bayes
- only 1 model, using same dataset as 1. lightgbm

## Second Stage - 2 MLPRegressors

I tried different ensembles of different groups of base models. These were my findings:

Average of 3 models of the same parameters scaled with factor n=1,2,4 for the public lightgbms (stage 1, models 3 and 4) worked pretty well.

Average of "few better models" and their ensemble showed better CV, but LB improvements didn't correspond.

This led me to build ensembles using all the models I built, even Naive Bayes with CV 0.55.

I tried different approaches - ElasticNet, BayesianRidge, LogisticRegression, My custom non-negative linear model (weights are either zero or positive, max 1, add up to 1), KNN, Lightgbm/XGboost and MLPRegressor.

I found out MLPRegressor works the best, so I ran a random gird search for 100 models: randomly choose number of hidden layers from 1 to 3, for each layer select randomly number of neurons up to 100.

Since MLPRegressors are non-linear, I tried not only single best models, but also average of few best MLPRegressors.

My final 2 second stage models were 2 MLPRegressors both with 2 hidden layers, first one had 52 and 94 neurones in the hidden layers, second one had 10 and 20 neurones.


## Third Stage - simple average

I averaged the outputs of the second stage models.

I saw discrepancy between my CV and LB (higher CV had lower LB score) so I computed CV two ways - mean of the 5 scores of each split: *CV1*, and single out-of-fold score, where probabilities where min/max scaled per fold: *CV2*.

Firstly I selected best LB submission with CV1 0.79861 and CV2 0.79871, which was the better out of the two final submissions, with 0.80833 private and 0.80074 public score.

For second submission I conservatively selected submission with the highest and most similar CV computed both ways with CV1 0.79928 and CV2 0.79924 which scored 0.80814 on private and 0.8004 on public LB.

My best submission which I didn't select with CV1 0.79947 and CV2 0.79871 scored 0.80849 on private and 0.80060 on public and was exactly the same as the described solution, with one more MLPRegressor with single hidden layer of size 54.
