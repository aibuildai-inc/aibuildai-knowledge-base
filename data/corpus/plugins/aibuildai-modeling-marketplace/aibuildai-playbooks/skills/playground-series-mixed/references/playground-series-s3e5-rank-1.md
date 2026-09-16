# [1st place solution] Single model (RAPIDS XGBoost)

Competition: playground-series-s3e5
Rank: #1
Source: https://www.kaggle.com/c/playground-series-s3e5/discussion/387882

Hello everyone! 

First, I would like to thank the Kaggle team for the competition. I have been learning a lot in these competitions, which is helping me sharpen my skills!

Unlike other solutions I saw in the Discussion for this competition, I did not use an ensemble. My solution was quite simple but effective (single xgboost model). I worked hard on the first two days of the competition to build a good baseline with a good CV.

## [Link to the notebook](https://www.kaggle.com/code/rapela/tpss3e5-1st-place-solution-rapids-xgboost)  🔥🔥

- Enjoy  :)


## Additional Data ❌

Original Whine or just the competition dataset? 

For me, this point was hard to decide. During the competition, I saw myself going down on the public leaderboard, but in the end, I decided not to use the additional dataset because it was overfitting a lot on my local CV.

## Training and Validation (CV) ✔️

For the CV, I used **StratifiedKFold** due to the imbalance, and I tuned the K based on some submissions to get the public score and check with my local CV. I started with K=5, but in the end, I saw that K=10 was more reliable with my experiments, then **K=10** was my final hparam.

##  Feature Engineering  (FE) ❌

I tried different FE and the ones available in some public notebooks, but my local CV was doing worse, so I removed it. In the end, my final model did not have a special FE.

## Model (RAPIDS XGBoost - GPU) 🔥🔥🔥

I always start with some standard models, like lgbm, xgboost, or catboost. For this competition, I wanted to have the best model as soon as possible because I wanted to iterate fast, so I started with the ** RAPIDS XGBoost**. For training, I used Kaggle GPUs (Thanks!). (xgb objective 'objective': 'reg:squarederror', 'tree_method': 'gpu_hist', early_stopping_rounds=50, 'num_boost_round': 1000). For the test set, I used the ntree_limit=model.best_iteration. The others hparams I will provide the others when I have time to clean the code and release the notebook! :)

## Regression Optimise Class Cutoff 💯

I would like to thank the discussions and public code available that I used to build my solution. One that I remember was the [Regression_OptimiseClassCutoff](https://www.kaggle.com/code/paddykb/ps-s3e5-regression-optimise-class-cutoff) from @paddykb and [https://www.kaggle.com/competitions/playground-series-s3e5/discussion/382525](https://www.kaggle.com/competitions/playground-series-s3e5/discussion/382525) from @abhishek (discussion: @carlmcbrideellis ) (I used the class OptimizedRounder doing my hyperparameter tuning). As the data distribution shift on my studies was not that high between train/test, I fit for every fold the optimizerRound on the validation predictions to get the cutoff for the test set. After the division by the number of folds, the predictions were float, so I used .round().astype(int) before submitting.

## Hyperparameters Tuning (Optuna) ✔️

For tuning the model, I used **Optuna** (where I played a lot with the ranges of the hparams and the number of trials). The metric that I was optimizing was cohen_kappa_score(weights='quadratic') on my train oof after the cutoff.

## Credits 🙏🙌

I would like to give credit to other competitors that worked really hard on the competition and the ones that shared a lot of content on the notebooks and discussion. Sorry if I forgot to give credit to someone, but I was not looking anymore at the competition since I was upset about the people using the external data and performing better on the public LB 😆

## Code Release ✨

I am planning to release the code when I have some time to clean it and prepare the notebook. That's it right now.

## TL;DR 💥

**- RAPIDS:XGBOOST + SKFold10 + Optuna + Regression Class Cutoff (I trusted my local CV to select the two final models).**
