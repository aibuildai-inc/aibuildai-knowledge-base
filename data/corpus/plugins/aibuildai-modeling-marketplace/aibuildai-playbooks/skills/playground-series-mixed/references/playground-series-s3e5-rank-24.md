# 24th Solution.

Competition: playground-series-s3e5
Rank: #24
Source: https://www.kaggle.com/c/playground-series-s3e5/discussion/386789

Hello,

Congratulations to the winners of this competition! 🙌

**Code**
Submit: https://www.kaggle.com/code/davidhguerrero/drop-features-reduction-with-t-sne-model?scriptVersionId=119026084    

**Overview**
First and foremost [Is this competition a lottery?](https://www.kaggle.com/competitions/playground-series-s3e5/discussion/383429) by [ambrosm](https://www.kaggle.com/ambrosm) 

Yes, of course to me 😊 



Simple solution:
Stacking Algorithms and Explore best K-Fold with Optuna without no more params of each algorithm. Agregatted original + syntetic dataset
Model A: LGBMClassifier
Model B: CatBoostClassifier
Model C: LGBMRegressorWithRounder



Train many quick models with Optuna (explore fewer params). 


First only  random_state &  n_splits of k-Fold.

```
scores =[]

def find_out_params_model(trial):
    random_state = trial.suggest_int('random_state', 1000, 2000)
    n_splits = trial.suggest_int('n_splits', 8, 20)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=conf.random)
    my_model = LGBMClassifier( 
        random_state = random_state
    )
    for fold, (train_idx, valid_idx) in enumerate(cv.split(X, y)):
        X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
        y_train , y_valid = y.iloc[train_idx] , y.iloc[valid_idx]
        my_model.fit(
            X_train, y_train,
            eval_set= [(X_valid,y_valid)],
            early_stopping_rounds = 50,
            verbose=0
        )
        
        preds_valid = my_model.predict(X_valid)
        score = cohen_kappa_score(y_valid,  preds_valid, weights = "quadratic")
        scores.append(score)
    return np.mean(scores)
```


And this class let me clean & organizate the notebook code
```
class conf:
    index = 'Id'
    target = 'quality'
    random = 2023
    
    load_original = True
    only_positive = False

    include_optuna = False
    
    include_lgbm = False
    include_catboost = False
    include_lgbm_regression = True
    n_trials = 10

np.random.seed(conf.random)
```
Measures the dependency between the variables with Mutual information add and remove some [Feature Engineering Ideas](https://www.kaggle.com/competitions/playground-series-s3e5/discussion/382698) thanks to [Jose Cáliz](https://www.kaggle.com/jcaliz)
Remove Model A & B. Better score with only one: LGBMRegressorWithRounder. Set a side other ones models

Esemble best models LGBMRegressorWithRounder 
```
if conf.include_lgbm_regression:
    scores = []
    for train_index, val_index in LGB_skf.split(X, y):
        x_train, x_val = X.iloc[train_index], X.iloc[val_index]
        y_train, y_val = y.iloc[train_index], y.iloc[val_index]

        m = LGBMRegressorWithRounder(**LGReg_best_param)
        m.fit(x_train, y_train, verbose = False)

        models.append(m)
        scores.append(cohen_kappa_score(y_val, m.predict(x_val), weights = "quadratic"))
    print(f'mean score: {np.mean(scores):.4f}')`
```
and check confidence of model 

`submission['quality'].value_counts()` vs `train['quality'].value_counts()`

**What  Worked**
- [High Impact]: 
 - [Transform Your Regressor with Rounder Integration](https://www.kaggle.com/competitions/playground-series-s3e5/discussion/382960) and [Feature Engineerhttps://www.kaggle.com/competitions/playground-series-s3e5/discussion/383429g Ideas](https://www.kaggle.com/competitions/playground-series-s3e5/discussion/382698) thanks to [Jose Cáliz](https://www.kaggle.com/jcaliz) 
 - Drop the features that provide no useful information mutual_info_classif + Add Features.
 - Explore with Optuna ONLY with random_state and num of Kfolds.
 - Set a side models with bad with scores Model A: LGBMClassifier & Model B: CatBoostClassifier.
 - Check a few other hyper params gets better score and add to model.

**What Didn't Work**
- Dimensionality Reduction to visualize features (high-dimensional9 data with TSNE to Aggregate this features into promising new features to train data set.
- Feature scaling: standardize or normalize features get worse score.



**Thanks and Acknowledgements**
Thanks again to all the competitors and to the organizers at Kaggle for another fun round in the Playground Series. 😄
