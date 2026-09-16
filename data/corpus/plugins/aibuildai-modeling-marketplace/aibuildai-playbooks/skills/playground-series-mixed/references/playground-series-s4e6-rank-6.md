# 6th Place Solution: Many model ensembles were detrimental?

Competition: playground-series-s4e6
Rank: #6
Source: https://www.kaggle.com/c/playground-series-s4e6/discussion/515989

Hi everyone,

My solution is simple: 

- 5-fold `StratifiedKFold` for validation.
- Included the original dataset.
- Ensemble of XGB and LGBM.
- Optuna hyperparamter  tuning.

I didn't dedicate as much time to this competition as I usually would because I noticed a significant amount of randomness in the outcomes. My assessment appears to be validated due to a new Kaggler @furgalhachaimajhi achieving 2nd place (congratulations btw!) with only a single submission.

The most interesting aspect of this competition for me was that single models or very small ensembles seemed to perform exceptionally well. It reminds of a Playground competition over a year ago where I achieved 54th place with a single CatBoost model. [Single Model Catboost PS S3 E4](https://www.kaggle.com/code/mattop/single-model-catboost-ps-s3-e4)

My solution was heavily inspired by @rzatemizel's notebook [LGBM + CATB+ XGB + NN: Voting or Stacking?](https://www.kaggle.com/code/rzatemizel/lgbm-catb-xgb-nn-voting-or-stacking) (make sure to upvote) which employs recursive feature elimination with cross-validation to determine the models to include in the ensemble.
