# 2nd Place - GBDT + NN + SVR + Original Data

Competition: playground-series-s5e3
Rank: #2
Source: https://www.kaggle.com/c/playground-series-s5e3/discussion/571176

Wow, what a shakeup! I was afraid of a shakeup from day one, so I kept my solution simple. 

# Tabular Data
In general with tabular data, I like to blend GBDT and NN. Then I try adding a few ML models like SVR, LR, KNN, etc. Furthermore in Kaggle playground competitions, we must decide how to use the original dataset that synthetic data was created from.

# Feature Engineering
When train data is small (few rows) I do little or no feature engineering because it is easy to overfit train data. When data is large (many rows like December and February playground comps), I do lots of feature engineering. 

In this competition, I chose to do no feature engineering. My solution is just an equal average of multiple models where each model trains on the data "as is" without feature engineering. So in this competition, I spent my time training different diverse models (using original data in different ways). And evaluating local ensemble OOF CV scores using Group K Fold. (And each ensemble uses equal weight averaging to avoid ensemble overfit).

# Group K Fold
In this competition, I used Group K Fold with 6 folds. I split the train data into 6 years and put each year in its own fold using Group K Fold. Because the test data is two new years of data

    train['group'] = train['id']//365

# Original Data as New Rows
The train.csv data is 6 years and 2190 rows. The original dataset is 1 year and 366 rows. One way to add the original data is `pd.concat()` and add **new rows**.  (It then becomes `group=7` for training and is ignored in the validation score calculation).

    train = pd.concat([train,orig],axis=1)

### => XGBoost - CV 0.893, Public LB 0.848, Private LB 0.90317
Single model uses `max_depth=3`, `colsample_bytree=0.9`, `subsample=0.9` like version 1 of my XGB starter notebook [here][1]. We use data "as is" without feature engineering. (Uses train data with orig as new rows).
### => TabPFN - CV 0.894, Public LB 0.867, Private LB 0.90193
Single model uses data "as is" without feature engineering. (Uses train data with orig as new rows).
### => [Two Model Blend] - CV 0.897, Public 0.859, Private LB 0.90474 - [11th Place]
This equal weight ensemble of two models above achieves 11th Place!

# Original Data as New Columns
The train.csv data has 11 feature columns. The original data as 11 feature columns. One way to add the original data is `pd.merge()` and add **new columns**. (We shared this idea last playground comp [here][3])
 
    m = train.rainfall.mean()
    for c in COLS:
        n = f"{c}2"
        train[n] = train[c].map( orig.groupby(c).rainfall.mean() )
        train[n] = train[n].fillna(m)

### => RAPIDS SVC - CV 0.896, Public 0.852, Private 0.90610 - [2nd Place]
Single model uses RAPIDS SVC with `C=0.1`, `kernel='poly'`, `degree=1` similar to my starter notebook [here][2]. We use data "as is" without feature engineering.  (Uses train data with orig as new columns). This single model achieves 2nd Place!
### => [Three Model Blend] - CV 0.898, Public 0.855, Private LB 0.90728 - [1st Place]
This equal weight ensemble of three models above achieves 1st Place!

# My Final 2 Submissions
For my final 2 submissions, I trained a few other models (CatBoost, LogisticRegression, XGBoost, SVR) and submitted two different 6 model equal weight blends. The three models above are the strongest. The additional models boosted ensemble CV score to `0.900` and `0.901` but did not boost private LB score beyond `0.906`. My final 2 submission 6 model ensembles were:
### => [Six Model Blends] - [2nd Place]
* CV = 0.900, Public LB = 0.857, Private = 0.90604
* CV = 0.901, Public LB = 0.867, Private = 0.90599

[1]: https://www.kaggle.com/code/cdeotte/xgboost-starter-ensemble-lb-0-935-wow
[2]: https://www.kaggle.com/code/cdeotte/rapids-svc-w-feature-engineering-lb-0-856
[3]: https://www.kaggle.com/competitions/playground-series-s5e2/discussion/565539
