# 5th Place Solution

Competition: playground-series-s3e2
Rank: #5
Source: https://www.kaggle.com/c/playground-series-s3e2/discussion/378780

Hi all,

It seems there was quite a good shake up given that dataset was highly imbalanced and AUC can vary a lot depending the number of samples. I realized there was a good difference between by OOF AUC and the leaderboard so I decided to trust only my CV (10 StratifiedKfold).

### Tricks that worked

1. Fill `unknown` category form `smoking status` as `never smoked`. The ituition was given on my **[EDA](https://www.kaggle.com/code/jcaliz/ps-s03e02-eda-baseline)** where you can see that `unknown` class has the lowest probability of stroke.
2. Fill `other` class from `gender` as `male`. I spotted a boost on CV when filling that record in synthetic dataset. I didn't probe the leaderboard to validate this on test.
3. Ensemble using gradient descent and ranking the predictions.
4. Concat original stroke dataset and use StratifiedKfold where validation only has synthetic data.
5. Feature selection using RecursiveFeatureElimanation. Additional features I tried:
```Python
def generate_features(df):
    df['age/bmi'] = df.age / df.bmi
    df['age*bmi'] = df.age * df.bmi
    df['bmi/prime'] = df.bmi / 25
    df['obesity'] = df.avg_glucose_level * df.bmi / 1000
    df['blood_heart']= df.hypertension*df.heart_disease
    return df
```

### Things that didn't work
1. Use forward selection taken from this [notebook](https://www.kaggle.com/code/cdeotte/forward-selection-oof-ensemble-0-942-private/notebook). This was my second submission and scored 0.89941 on private leaderboard. I think It didn't worked because the final ensemble was only composed of XGBoost models while my best submission has a wide variety of models.
2. MeanEncoder, WoEEncoder and CountFrequency encoder. Neither of those provided better solutions that OneHotEncoder.

### Final Ensemble:
My final ensemble is composed of several models:
* LogisticRegression with RFE, l2, and liblinear solver.
* LogisticRegression with RFE, no regularization, lbfgs solver.
* LightGBM no RFE, no Feature Engineering.
* Another LightGBM with early stopping and monitoring logloss (yes, logloss no AUC).
* A Catboost model inspired in [this notebook](https://www.kaggle.com/code/dmitryuarov/ps-s3e2-catboost-lasso-nn-fiasco) by @dmitryuarov. I made some modifications to make sure the OOF AUC was similar to the mean AUC by fold.
* A tuned XGBoost with feature engineering. (best single model) See the code and results replica **[Here](https://www.kaggle.com/code/jcaliz/ps-s03e02-eda-baseline)**

And that's all.
Many congratulations to the winners, looking forward to the next playground competitions.
