# My 9th place solution to the September Tabular competition

Competition: tabular-playground-series-sep-2021
Rank: #9
Source: https://www.kaggle.com/c/tabular-playground-series-sep-2021/discussion/276248

Hi fellow kagglers,

Coming from the 30 days of ML competition, this is my first competition on a classification problem.
I'd like to summarize quickly what I have done in this competition, things that have worked and others.

## How my score evolved compared with the best final score of the competition

As a visual aid, here is the evolution of my score on the private LB. More exactly, I plot the difference between the best score of the competition (from @pourchot) and mine, everything in log scale, to better illustrate the slow journey of model improvement (symbols correspond to different notebooks aka different approaches, color to versions of notebooks):


## Initial models

Shortly after the competition started, @prikshitsingla found that the number of nans in a row is a good feature for predicting target and indeed, it’s been shown to be correlated the target value ([post](https://www.kaggle.com/c/tabular-playground-series-sep-2021/discussion/270206)). This allowed me to elevate my auc score above a sound 0.8. Amazing. A good understanding of data is indeed crucial for building models on them. I've not done much EDA in this competition and I hope I can improve on that in the future.

Then as always, I began to throw in xgboost and lightgbm classifiers and used Optuna to tune hyperparameters. Due to the large size of the data file, I used random sampling to work on 20% of the whole data. This has allowed me to run Optuna quickly, with 5-fold cross validation, and quickly zoomed in on a combination of hyperparameters that worked quite well. Unfortunately, further Optuna searches led to nowhere and gave no improvement on the public LB score. All in all I was not quite successful in uncovering a lot of great models.. so I began to refer to works from others and use stacking.

## Stacking

Then began the stacking part. The principle is well explained by @abhishek22211 in [his post](https://www.kaggle.com/c/30-days-of-ml/discussion/265755). Basically one needs to use cross-validations on a number of so called level-0 models, then use their out-of-fold predictions to train a level-1 meta model for a final prediction.

Thanks to @vishwas21 ([notebook](https://www.kaggle.com/vishwas21/tps-sep-21-3-level-custom-stacking)), @manabendrarout ([notebook](https://www.kaggle.com/manabendrarout/custom-stacking-of-classifiers-gpu-tps-sep2021/)) and @mlanhenke ([notebook](https://www.kaggle.com/mlanhenke/tps-09-simple-blend-stacking-xgb-lgbm-catb/)), I was able to gather 34 models in total at my L0 level.

In terms of stacking, what worked really well were:
- use LinearRegression instead of logistic reg as L1 meta model.
- Reuse L1 model's and L0 models' results as inputs to L2 meta model, which is again a LinearRegression.

I also did some target encoding based on number of nans in a row, and used various seeds to replicate some models, which seemed to work. Also thanks to @edrickkesuma for the excellent [post](https://www.kaggle.com/c/tabular-playground-series-sep-2021/discussion/273253) on power blending, I’ve learned some new stuff !

What didn't help my stacking score:
- xgboost or lightgbm classifiers as meta model. and using Optuna to tune them at L1 level.
- blindly add models at L0 level with mediocre scores just for diversity's sake (RF, LogisticRegression, etc.)
- pseudo labeling ([post](https://www.kaggle.com/c/tabular-playground-series-aug-2021/discussion/270051)).

After this I was about halfway into my climb towards my final ranking, at about 30th position on the public LB.

## “One model voting” and “Fit to all”

Then thanks to @martynovandrey's [post](https://www.kaggle.com/c/tabular-playground-series-sep-2021/discussion/274404), I began to replicate lightgbm models from 
@hiro5299834 ([notebook](https://www.kaggle.com/hiro5299834/tps-sep-2021-single-lgbm)), @ivankontic ([notebook](https://www.kaggle.com/ivankontic/004-2o-lightgbm-colsample-tps-sep-2021)), @mlanhenke ([notebook](https://www.kaggle.com/mlanhenke/tps-09-simple-blend-stacking-xgb-lgbm-catb)) and @realtimshady ([notebook](https://www.kaggle.com/realtimshady/single-simple-lightgbm)), plus a xgboost model of my own, each with 7 different random_state seeds and blending, which all gave a noticeable improvement of score (Although in hindsight, the resulting rise in the public LB is more pronounced than the rise in the private LB. So the "effectiveness" of these improvements may be less than, say, what stacking models gives.).

A further improvement has been achieved by fitting a model to all train data and to prolonging the value of `n_estimators` to go beyond its 'optimal' value found in a cross-validation with early stopping. The rationale here being that 
- by training a 'fit to all' model on the whole data, it should be possible to push its complexity further to learn better without falling into overfitting, hopefully.
- And sometimes the early stopping may have been activated too early in the cv.

This may be the most important thing that I've learned in this competition. In fact, I was quite used to rely fully on the cross-validation for test data prediction, that is to say:
- use Kfold or StratifiedKFold to separate the whole training data into K folds, with `early_stopping_rounds` around 300-500.
- At each CV iteration, train a new model using non-validation folds, check score on the validation fold, and use this model to predict on the test data.
- Blend all predictions on the test data to give the final prediction.

Although there should be some benefit here by using a blended prediction from all cv iterations, one should also consider that by doing this not one model has seen and been trained on the whole training data, which may be a loss of opportunity. **So the bottom line is, trust the hyperparameters validated using cross-validation, but do give a check later on the `n_estimators` by training on the whole data to see if any improvement can be achieved on the public LB.**

Finally, I blended my L1 prediction from my stacking notebook, "fit to all" predictions using 5 models and the impressive submission from @mlanhenke ([notebook](https://www.kaggle.com/mlanhenke/tps-09-simple-blend-stacking-xgb-lgbm-catb)) for my final submission. To compare all the predictions, [the density plot](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.plot.kde.html) of their predicted values looks like this

There are indeed some models that gave more eccentric results than others. I thought about using different wights / power blending etc here but decided not to, and in the end, it has worked rather well and put me among the top 10 on the private LB. 😊

Thanks for reading. If you have questions please leave a message. Keep kaggling and see you in future competitions !
