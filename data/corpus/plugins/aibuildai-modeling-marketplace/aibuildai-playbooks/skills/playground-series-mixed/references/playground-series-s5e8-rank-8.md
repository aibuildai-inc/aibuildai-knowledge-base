# 8th place - hill climb selected meta-learners

Competition: playground-series-s5e8
Rank: #8
Source: https://www.kaggle.com/c/playground-series-s5e8/writeups/8th-place-hill-climb-selected-meta-learners

Inspired by the [5 submission challenge](https://www.kaggle.com/competitions/playground-series-s5e8/discussion/593993) by @yunsuxiaozi, I decided to limit my submissions to the number of days in the month (31). This seemed enough for a few experiments before focussing on a specific approach. Will probably stick with this `submissions == no. of days in month` limit for future playground series I participate in.

## the solution

My approach was to train a diverse set of base models. These consisted of linear models, boosted trees, and neural nets. The base models where trained with different sets of features and tuned with different parameters. I then ran hill climbing on these base models. The selected models where used to train meta-learners. 
Note that the base models feed into the meta-learners where not weighted in any way. Any of the base models that received a positive weight from hill climbing we're used as input into meta-learners. 
I then ran hill climb on the results of these meta-learners to get the finial submission.

Submission results:
- CV: 0.97742
- Public score: 0.97801
- Private score: 0.97768

The public score for both my selected submissions was 0.97801, their private score is also identical, and their CV differed in the last digit. 

```
Best ROC AUC Score: 0.97742733 - v7 | lb 0.97801 *
Best ROC AUC Score: 0.97742735 - v8 | lb 0.97801 *
```

I used the same cv split for all the base models
```
SEED = 208
FOLDS = 5
cv = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=SEED)
```

For the meta-learners I used `FOLDS = 10`.

### selected base models

```
Final ensemble weights (high → low):
  0.2490 - cdeotte xgboost - orig as columns
  0.1117 - ps-s5e8-xgboost-deep
  0.0905 - nn-by-gpt5
  0.0777 - CatBoostClassifier (ensemble ii)
  0.0714 - xgboost and nn ensemble
  0.0670 - CatBoostClassifier (ensemble iv)
  0.0577 - LGBMClassifier-params_v12
  0.0556 - LGBMClassifier-ii-TE-std
  0.0516 - CatBoostClassifier (ensemble)
  0.0353 - NeuralNetFastAI_BAG_L2-ii
  0.0324 - CatBoostClassifier (ensemble hist-ii)
  0.0284 - cdeotte xgboost - ensemble
  0.0262 - SGDClassifier (stack light-ii)
  0.0169 - RandomForestEntr_BAG_L2-ii
  0.0117 - RandomForestClassifier (ensemble ii)
  0.0098 - RandomForestEntr_BAG_L2-iii
  0.0053 - RandomForestClassifier (ensemble)
  0.0009 - HistGradientBoostingClassifier (ensemble histbook)
  0.0009 - cdeotte xgboost - orig as rows
```

### final meta-learners selection

```
Final ensemble weights (high → low):
  0.223215 - WeightedEnsemble_L2-l2
  0.177151 - NeuralNetTorch_r79_BAG_L1-l2
  0.142211 - NeuralNetTorch_BAG_L1-l2
  0.101949 - LightGBM_r131_BAG_L1-l2
  0.067658 - CatBoostClassifier (l2 boruta)
  0.066139 - cdeotte xgboost more - orig as columns
  0.064272 - nn-by-gpt5-more
  0.057201 - NeuralNetFastAI_BAG_L1-l2
  0.044992 - LightGBMXT_BAG_L1-l2
  0.029317 - XGBoost_BAG_L1-l2
  0.014647 - NeuralNetFastAI_r191_BAG_L1-l2
  0.008044 - RandomForestGini_BAG_L2-iii-more
  0.002062 - LightGBM_BAG_L1-iii-more
  0.000923 - XGBoost_BAG_L1-iii-more
  0.000220 - HistGradientBoostingClassifier (l2 boruta)
```

Note that here "cdeotte xgboost more - orig as columns" and "nn-by-gpt5-more" are models by @cdeotte trained on the oofs of the selected base models. These were adapted to match my own cv split and do regular training (without full fit).

## feature engineering

I used a combination of the following functions depending on the model type.

```python
def feature_engineer(df):
    df['has_debt'] = (df['balance'] < 0).astype(int)
    df['long_duration'] = (df['duration'] > 300).astype('category')
    df['duration_sqrt'] = np.sqrt(df['duration']).astype('float32')
    df['duration_log'] = np.log1p(df['duration'])
    df['duration_sin'] = np.sin(2*np.pi * df['duration'] / 540).astype('float32')
    df['duration_cos'] = np.cos(2*np.pi * df['duration'] / 540).astype('float32')
    df['balance_log'] = (np.sign(df['balance']) * np.log1p(np.abs(df['balance']))).astype('float32')
    df['balance_sin'] = np.sin(2*np.pi * df['balance'] / 1000).astype('float32')
    df['balance_cos'] = np.cos(2*np.pi * df['balance'] / 1000).astype('float32')
    df['age_sin'] = np.sin(2*np.pi * df['age'] / 10).astype('float32')
    df['pdays_sin'] = np.sin(2*np.pi * df['pdays'] / 7).astype('float32')

    df['duration_bin_20'] = pd.qcut(df['duration'], q=20, labels=False, duplicates='drop')
    df['balance_bin_20'] = pd.qcut(df['balance'], q=20, labels=False, duplicates='drop')

    # ref: https://www.kaggle.com/code/ganeshataqwa/0-96-classifying-bank-customers-let-s-do-it
    df['is_first_contact'] = np.where(df['pdays'] == -1, 1, 0)
    df['contact_ratio'] = df['campaign'] / (df['previous'] + 1)
    df['economic_stability'] = df['balance'] / df['age']
    high_months = ['mar', 'oct', 'sep', 'dec']
    df['is_high_conversion_month'] = np.where(df['month'].isin(high_months), 1, 0)
    df['is_short_call'] = np.where(df['duration'] <= 150, 1, 0)

    month_map = {
        'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6, 
        'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12
    }
    df['month_num'] = df['month'].map(month_map).astype(int)
    df['month_sin'] = np.sin(2*np.pi * df['month_num'] / 12).astype('float32')
    
    df = df.drop(['month_num'], axis=1)
    return df
```

Create combinations of categorical features.

```python
def pairwise_combinations(train, test, to_combine):
    encoded_columns = []
    pair_size = [2, 3]
    
    for r in pair_size:
        for cols in tqdm(list(combinations(to_combine, r))):
            col_name = '_'.join(cols)
            
            train[col_name] = train[list(cols)].astype(str).agg('_'.join, axis=1)
            train[col_name] = train[col_name].astype('category')
            
            test[col_name] = test[list(cols)].astype(str).agg('_'.join, axis=1)
            test[col_name] = test[col_name].astype('category')
    
            encoded_columns.append(col_name)
    
    print(len(encoded_columns), 'new features added')
    return train, test

to_combine = ['default', 'housing', 'loan', 'poutcome', 'balance', 'duration', 'previous']
X, X_test = pairwise_combinations(X, X_test, to_combine)
```

Interaction of numeric features

```python
def add_interaction_features(df, features):
    data = df.copy()
    for f1, f2 in itertools.combinations(features, 2):
        data[f'{f1}_plus_{f2}'] = data[f1] + data[f2]
        data[f'{f1}_minus_{f2}'] = data[f1] - data[f2]
        data[f'{f1}_div_{f2}'] = data[f1] / (data[f2] + 1e-5)
        data[f'{f1}_times_{f2}'] = data[f1] * data[f2]
    return data

nums = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']
X = add_interaction_features(X, nums)
X_test = add_interaction_features(X_test, nums)
```


## how i used original data

I used the original data in multiple ways.  
The first was suggested in this [comment](https://www.kaggle.com/competitions/playground-series-s5e8/discussion/596178#3261839) by @siukeitin. Specifically the combined augmentation and post processing.

```
model = Augmented(
    Postprocessed(LGBMClassifier, contrarian)(**light_params_v28), X_orig, y_orig
)
```

I also used the following function discussed by @jmascacibar here: [The original dilemma](https://www.kaggle.com/competitions/playground-series-s5e8/discussion/597903)

```python
def add_original_cols(df_train, df_test, df_orig, feats, target_col='y'):
    '''
    Add original features groupby original target to the synthetic data
    ref: https://www.kaggle.com/competitions/playground-series-s5e8/discussion/597903
    '''
    train = df_train.copy()
    test = df_test.copy()
    tm = df_orig[target_col].mean()
    add_feats = []
    for feat in feats:
        if feat in df_orig.columns:
            name = f'{feat}_orig_target_mean'
            mapping = df_orig.groupby(feat)[target_col].mean()
            train[name] = train[feat].map(mapping)
            train[name] = train[name].fillna(tm)
            test[name] = test[feat].map(mapping)
            test[name] = test[name].fillna(tm)
            add_feats.append(name)
            print(f'Added {name} feature')
    print('\n---- Complete ----\n')
    print(f'Train, Test shape: {train.shape, test.shape}')
    return train, test
```

Finally, the only public models I used in my base models where from @cdeotte, see his discussion posts for more info
- [XGBoost - QuantileDMatrix Trick](https://www.kaggle.com/competitions/playground-series-s5e8/discussion/600048)
- [NN Starter Notebook - Vibe Coding GPT5 - CV 0.974](https://www.kaggle.com/competitions/playground-series-s5e8/discussion/600617)


## conclusion

- Ignore the blind blenders.
- Focus on CV not LB.
- Learn by incorporating ideas into your own work, not just duplicating them.
- Learn by reading high quality posts from previous contests.


This is my first write up so let me know if anything isn't clear and feel free to ask about anything I didn't cover.

That's it, all the best 🖖🏾.
