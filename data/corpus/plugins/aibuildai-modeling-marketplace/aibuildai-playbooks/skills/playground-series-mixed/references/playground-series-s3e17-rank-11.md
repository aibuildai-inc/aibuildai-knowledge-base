# 11th Place Solution | Unexpected Top 1%

Competition: playground-series-s3e17
Rank: #11
Source: https://www.kaggle.com/c/playground-series-s3e17/discussion/419643

I already expected that the top places will get few shake-up but getting myself up to top 1% despite telling all my approaches? That's different story. Anyway, here's what I've done to get to 11th position

# Cross-Validation Process

I used MultilabelStratifiedKFold from iterative-stratification library because I want to make sure that the percentage for all types of failure is kept throughout the fold. I also explained this [here](https://www.kaggle.com/competitions/playground-series-s3e17/discussion/416904). The amount of split I used was 10 folds, because 5 folds isn't correlated enough with the public LB.

I always try to do everything inside cross-validation pipeline, such as adding the original data, encoding, scaling, etc. This way, I don't have to worry a lot about leakage.

# Feature Engineering

I use category-encoders' CatBoost Encoder in all my models, except CatBoost itself, since I noticed that CatBoost performs exceptionally well for some reason. I've already posted about this [here](https://www.kaggle.com/competitions/playground-series-s3e17/discussion/419111) actually, the main difference is that I put it inside the model pipeline instead of using it before doing a CV, thus preventing major leakage. Example code is as follows:

```python
Encoder = CatBoostEncoder(cols = ['Product ID', 'Type'])
model = make_pipeline(Encoder, model)
```

As for creating new features, I've only created one: `IsFailure`. This feature describes whether there is any type of failure (TWF, HDF, etc.) that is happening. I only use this feature in one model: Gaussian Naive Bayes. The rest is just encoding.

# Tuning and Ensembling

I used 6 models: Gaussian Naive Bayes, Random Forest, XGBoost, LightGBM, LightGBM's Dart, and CatBoost. For gradient boosting models, I used Optuna to tune them. For Naive Bayes and Random Forest, I didn't tune them at all, only calibrated them (I also posted about it [here](https://www.kaggle.com/competitions/playground-series-s3e17/discussion/419432)), and did further preprocessing for Naive Bayes (such as scale normalization and Yeo-Johnson transformation).

Once I've done building all of them, I used LogisticRegression to find optimal weight for my voting ensemble. The CV score is 0.98006. Finally, I retrained my ensemble on the whole dataset for better final result instead of just relying on the CV process.

Edit: I just realized that my best submission didn't even use Random Forest and LightGBM's DART

Edit 2: Full code can be found here:
https://www.kaggle.com/code/iqbalsyahakbar/11th-place-solution-code
