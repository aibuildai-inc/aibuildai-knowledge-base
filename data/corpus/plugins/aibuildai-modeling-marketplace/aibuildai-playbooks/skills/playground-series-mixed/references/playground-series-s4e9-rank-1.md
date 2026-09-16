# # 1 solution - stacked NN

Competition: playground-series-s4e9
Rank: #1
Source: https://www.kaggle.com/c/playground-series-s4e9/discussion/537052

Did I really made it to the top? I am still surprised and excited.

**Way to the solution:** I spent the first two weeks with reading the discussions, playing around with catboost and publishing an ensemble. The plan was to collect diverse models and ensemble them with Ridge, with the same pipeline I used in [this notebook](https://www.kaggle.com/code/martinapreusse/ps4e9-cat-svr-lgbm-nn-py). The final ensemble, which I chose as my first final submission, would have landed me on the second place and differed from the notebook in the following points:
- I used 20 cv folds
- I included original data in some of the models (even two times in LGBM)
- I did compute a SVR with a rbf kernel as suggested by [broccoli beef](https://www.kaggle.com/siukeitin) in [this discussion post](https://www.kaggle.com/competitions/playground-series-s4e9/discussion/532997) instead of the linear SVR.
- I included all categorical features additionally as target encoded to catboost, but I used the median, not the mean for target encoding. I did this leakfree, meaning that I recomputed the targetencoded columns in each fold. Moreover, I used Catboost as classifier, not as regressor. Catboost predicted the outlier prices (see function `bin_price`). The hyperparameters were found by optuna. The oof predictions were not used in the ensemble. They were used as an additional feature in a LGBM (or the NN for my second final submission).

```python
def bin_price(data):
    df = data.copy()
    # Calculate Q1 (25th percentile) and Q3 (75th percentile)
    Q1 = np.percentile(df['price'], 25)
    Q3 = np.percentile(df['price'], 75)
    IQR = Q3 - Q1

    # Define the lower and upper bounds for outliers
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Identify outliers
    outliers = df[(df['price'] > upper_bound)]
    df['price_bin'] = (df['price'] < upper_bound).astype(int)
    
    return df

cat_params2 = {
    'early_stopping_rounds':25,
    'use_best_model': True,
    "verbose": False ,
    'cat_features': cat_cols,
    'min_data_in_leaf': 16, 
    'learning_rate': 0.03355311405703999, 
    'random_strength': 11.663619399375248, 
    'l2_leaf_reg': 17.703146378123996, 
    'max_depth': 10, 
    'subsample': 0.9479174100256215, 
     'border_count': 130, 
    'bagging_temperature': 24.032067560148384
}
```
- I included the catboost oof predictions as an additional feature for LGBM
- I used a second LGBM (LGBM5), where I label encoded all categorical data (rare categories summarized in a category "rare" as done with the NN in the notebook) and raised max_bin.
```python
lgb_params = {
    'verbose' : -1,
    'early_stopping_rounds':25,
    'loss_function':"RMSE",
    'n_estimators': 2000, 
    'max_bin': 30000,
}
```
- I included fastai computations from Autogluon (with a nested cv over 20 folds to be 100% leakfree)

```python
 predictor = TabularPredictor(label='price',
                             eval_metric='rmse',
                             problem_type="regression").fit(X_train,
                                                       pseudo_data = data_original, 
                                                       num_bag_folds = 10,
                                                       num_bag_sets = 2,
                                                       time_limit=1800,
                                                       included_model_types = ['FASTAI'], 
                                                       keep_only_best = True,
                                                       presets="best_quality",
                                                      )

```
I ended up with a crossvalidation score of 72300 and the following models (_st means that the catboost oofs are included):

The crossvalidation scores of the individual models were:


**First place solution**: I needed a second final submission and I decided spontanously on the last day to submit [a forked notebook ](https://www.kaggle.com/code/yekenot/ps-s4-e9-deeptables-nn-starter) from [Vladimir Demidov](https://www.kaggle.com/yekenot). I noticed that his NN is robust to changes, so I added four numerical features: the SVR oof predictions, the LGBM5 oof predictions, the CatboostClassifier oof predictions and XGB predictions (derived from publicy available hyperparameters, unfortunatly I forgot the source). The NN ensemble had a crossvalidation score of 72468, but in the end was better than the Ridge ensemble.

My thanks obviously go to @yekenot , @siukeitin , @noodl35 (LGBM hyperparameters), who directly provided parts of the code I used. I also provited strongly from the discussions, especially the AutoML solution threads and the posts from @tilii7 and @roberthatch where I got the idea for outlier classification and @cdeotte who made the entrance to NNs simple for me. I am also very grateful to all who are participating lively in the discussions so that learning is a fulfilling experience.

**What did not work:** I experimented with a lot of models, but most of them did not help me to get a better cross-validation score. Especially XGB did not work for me and although it is accidently included in my final submission I do not think that it was a crucial part of the ensemble.
Feature engineering was at least partly working, but all the amazing features introduced by Chris Deotte in [this post](https://www.kaggle.com/competitions/playground-series-s4e9/discussion/533961) did not work for me.

The corresponding crossvalidation - leaderboard scores in a scatterplot:


**Edit:** I included the CatBoostClassifier, LGBM1_st and LGBM5_st in [my notebook](https://www.kaggle.com/code/martinapreusse/ps4e9-cat-svr-lgbm-nn-py). This notebook has a private score of 62957.83525 and would have reached a place within the top three.

**Edit 2:** Autogluon, when using `TabularPredictor.fit()`, is not using the original data. To use the original data, one has to call `TabularPredictor.fit_pseudolabel()`. So my Fastai - AutoGluon predictions were not using the original data. For a nice reference of how to use AutoGluon with original data and with a predefined stratification see this great [notebook](https://www.kaggle.com/code/ravaghi/s04e10-loan-approval-prediction-autogluon/comments#3010042) of [Mahdi Ravaghi](https://www.kaggle.com/ravaghi) for the next playground competition.
