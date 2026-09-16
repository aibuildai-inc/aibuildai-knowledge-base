# 6th Place - A lot of features, a lot of ensembling

Competition: playground-series-s6e1
Rank: #6
Source: https://www.kaggle.com/c/playground-series-s6e1/writeups/6th-place-a-lot-of-features-a-lot-of-ensembling

First of all, thank you to Kaggle for organizing and to @cdeotte, @tilii7, @siukeitin , @ravi20076 and many more in the community for the discussions, code and ideas.

# Feature Engineering - Unsuccessful

As I said in [this discussion](https://www.kaggle.com/competitions/playground-series-s6e1/discussion/671022#3399439) I really tried to make feature engineering work for this competition.
The starter notebook for this is @cdeotte's excellent [notebook from March 2025 here](https://www.kaggle.com/code/cdeotte/first-place-single-model-lb-38-81).
I also highly recommend you read [the corresponding writeup.](https://www.kaggle.com/competitions/playground-series-s5e2/writeups/chris-deotte-1st-place-single-model-feature-engine)

I was largely unsuccessful with my best model achieving `8.59951/8.56195 Private LB/Public LB, 8.5950 CV`.
You can see my best model in [the notebook here.](https://www.kaggle.com/code/include4eto/xgb-nested-lr-nested-te-my-best-single-model#Model-Training)
Looking at other writeups, this seems to be competitive with what others have, so perhaps that's not too bad!

# Feature Engineering - Good for ensembling

I trained over `~200` XGB models this month with different features but I didn't actually spend much real-life time on individual models.
Rather I added whatever wacky feature ideas I had and let the models run, ensembling and looking at CV scores one-by-one.

I also sometimes took some of the best public notebooks and added them to my ensemble when I could, including:

* [Pseudo-labels + hillclimbers](https://www.kaggle.com/code/maiernator/pseudo-labels-hillclimbers), @maiernator 
* [Hillclimbers/RidgeCV stacking](https://www.kaggle.com/code/thomastschinkel/s6e1-8-54753-hill-climbing-ridgecv-stacking), @thomastschinkel 
* [TabM + FE](https://www.kaggle.com/code/omidbaghchehsaraei/tabm-predicting-student-test-scores-cv-8-61131), @omidbaghchehsaraei 
* [Autogluon](https://www.kaggle.com/code/omidbaghchehsaraei/autogluon-predicting-student-scores-lb-8-5767), @omidbaghchehsaraei 

The core observation here is - always save your OOF files and always try to use the same `KFold` split and seed - this lets you do a lot of things _after_ even if your feature engineering is unsuccessful.

My top feature was:

**Linear Regression**
As [discussed in detail here.](https://www.kaggle.com/competitions/playground-series-s6e1/discussion/665915), using a linear model helped a lot.
This helps us even more, as many features that don't help trees may help the regression, primarily:

- Monotonic transformations - trees care about ordering only, but the Linear Regression can use nonlinear monotonic transformations (e.g. `^2`, `log`) to improve its score!
Importantly, we use a triply nested-CV approach to avoid leakage:

```py
# loop 1 - outer fold
for i, (train_index, val_index) in enumerate(kf.split(train)):
    kf2 = KFold(n_splits=INNER_FOLDS, shuffle=True, random_state=42)
    
    lr_test_preds = np.zeros(X_test.shape[0])
    lr_val1_preds = np.zeros(X_val.shape[0])
    
    # loop 2 - inner fold
    for j, (train_index2, val_index2) in enumerate(kf2.split(X_train)):
        # loop 3 - target encoding categorical features
        target_encoder = TargetEncoder(...)

        lr_model = LinearRegression(n_jobs=-1)
        lr_model.fit(X_train2_enc, y_train2.ravel())

        lr_test_preds += lr_model.predict(X_test_enc)
        lr_val1_preds += lr_model.predict(X_val1_enc)
        X_train.loc[val_index2, 'lr_pred'] = lr_model.predict(X_val2_enc)
    
    X_test['lr_pred'] = lr_test_preds / INNER_FOLDS
    X_val['lr_pred'] = lr_val1_preds / INNER_FOLDS
```

You can see my best model using this approach in [the notebook here.](https://www.kaggle.com/code/include4eto/xgb-nested-lr-nested-te-my-best-single-model#Model-Training)

# Gating - Also Good for ensembling

My initial observation was that the data in the middle was roughly normally distributed:



As I mentioned in[ the discussion here](https://www.kaggle.com/competitions/playground-series-s6e1/discussion/671022#3399439), models have a higher CV score when trained only in the middle:

Mid range only (val is also midrange):
```
7 Folds:
Validation RMSE MID: 8.49981
Validation RMSE MID: 8.50462
Validation RMSE MID: 8.47966
Validation RMSE MID: 8.51556
Validation RMSE MID: 8.48685
Validation RMSE MID: 8.52697
```

Same 7 folds, all train (standard):

```
Validation RMSE: 8.56438
Validation RMSE: 8.60592
Validation RMSE: 8.60786
Validation RMSE: 8.58694
Validation RMSE: 8.62379
Validation RMSE: 8.58134
Validation RMSE: 8.62743
```

Naturally I thought we can train a classifier and "gate" the output of models where the classifier thinks we're at the limit (`19.599` and `100`):
Classifiers were quite poor at this, however:

```
                       precision    recall  f1-score   support

      Class 0 (<27.099)       0.56      0.43      0.49     19265
Class 1 (27.099-92.5)       0.95      0.97      0.96    571993
        Class 2 (>=92.5)       0.61      0.49      0.55     38742
 
             accuracy                           0.92    630000
              macro avg       0.71      0.63      0.66    630000
         weighted avg       0.92      0.92      0.92    630000
```

## Gating as post-processing

In the end I used gating as post-processing in ensembling, as [shown in my notebook here](https://www.kaggle.com/code/include4eto/min-max-gating-classifier-ensemble).

This uses linear regression twice, effectively "ensembling" linear regressions over linear regressions.
 Using K linear regressions as meta-predictors and then another linear transformation on top is still linear:

$$
y_{test} = (\sum_i X w_i) w = \sum_i (X w w_i) = \sum_i X (b_i)
$$

omitting the intercept, etc. - you can make this into a single affine transformation as well - it's still a linear combination of the OOF predictions.

This can really only be used as a trick for a slight CV/LB bump at the end.
The **downside** is that it can be leaky, if our OOF predictions use different folds per model.
The leakage didn't seem to matter for linear regressions this time, but for more complex models it will! The only thing is that the CV is now too optimistic once this is done:



I also used Gating as post-processing for individual models, adding them to my ensemble.
This seemed to work well for **bad models**, bumping their CV score slightly.

I did this with two models (one for left, one for right), which is necessarily **NOT** better (statistically), but helped my models achieve a better accuracy individually:

```py
# mix of 10 tresholds
df_oof = {}
df_test = {}

for p in np.linspace(0.75, 0.99, 50):
    MIN_P = p
    MAX_P = p
    
    _cond_left = (proba_oof_left >= MIN_P) & (proba_oof_right < 1e-3)
    _cond_right = (proba_oof_right >= MAX_P) & (proba_oof_left < 1e-3)
    gated_oof = np.where(_cond_left, MIN, best_model_oof)
    gated_oof = np.where(_cond_right, MAX, gated_oof)

    _cond_left = (proba_test_left >= MIN_P) & (proba_test_right < 1e-3)
    _cond_right = (proba_test_right >= MAX_P) & (proba_test_right < 1e-3)
    gated_test = np.where(_cond_left, MIN, best_model_test)
    gated_test = np.where(_cond_right, MAX, gated_test)
    
    df_test[str(p)] = gated_test
    df_oof[str(p)] = gated_oof

df_oof['non_gated'] = best_model_oof
df_oof = pd.DataFrame(df_oof)
df_oof.head()

df_test['non_gated'] = best_model_test
df_test = pd.DataFrame(df_test)
df_test.head()

# fit a regression on this
lr = LinearRegression(fit_intercept=False)
lr.fit(df_oof, train.exam_score)
```

This achieves in the order of `~0.01` improvement in score for individual models, but even more importantly **creates more diversity for ensembling**.

## Meta models

Finally, I also trained meta models on the probabilities and predictions from other XGB Models:

```
features = [
    [proba_left, proba_mid, proba_right, xgb_prediction]
]
```

# Ensemble

Finally, my ensemble consists of `231` models:


