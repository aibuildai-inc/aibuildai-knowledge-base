# 14 th solution (that could have been 6 th )

Competition: tabular-playground-series-aug-2022
Rank: #14
Source: https://www.kaggle.com/c/tabular-playground-series-aug-2022/discussion/349810

This is the first competition where my team mate @medali1992 and i reached the Top 1%. the  first thing we noticed was the difference  in the modalities between the measurements in the train set and those in the test set. This made us ensure to use these new modalites (test set ) in the validation process. The  3 vs 2 cross validation scheme by   [**AmbrosM**](https://www.kaggle.com/competitions/tabular-playground-series-aug-2022/discussion/341896) was **the key** to not **overfit** the leaderboard. 
we used this  code :
``` 
 folds_dict = {f'Fold 1': [['C', 'D', 'E'], ['A', 'B']], 
               'Fold 2': [['B', 'D', 'E'], ['A', 'C']],
               'Fold 3': [['B', 'C', 'E'], ['A', 'D']],
               'Fold 4': [['B', 'C', 'D'], ['A', 'E']],
               'Fold 5': [['A', 'D', 'E'], ['B', 'C']],
               'Fold 6': [['A', 'C', 'E'], ['B', 'D']],
               'Fold 7': [['A', 'C', 'D'], ['B', 'E']],
               'Fold 8': [['A', 'B', 'E'], ['C', 'D']],
               'Fold 9': [['A', 'B', 'D'], ['C', 'E']],
               'Fold 10': [['A', 'B', 'C'], ['D', 'E']]} 
 for fold in folds_dict.keys():
   print(f'########################## {fold} ##########################')
    
   x_train, y_train = df_train[df_train['product_code'].isin(folds_dict[fold][0])][features].values, 
   df_train[df_train['product_code'].isin(folds_dict[fold][0])]['failure'].values
   x_valid, y_valid = df_train[df_train['product_code'].isin(folds_dict[fold][1])][features].values, 
   df_train[df_train['product_code'].isin(folds_dict[fold][1])]['failure'].values
```
If you'd like a better implementation of this code do check the work of [ROBERT STOCKTON](https://www.kaggle.com/code/purist1024/principled-3-vs-2-cv-splitting-on-product-code) it's a more Sklearn way to write the same code.

 Our best single model is Tabnet, yet after the competition we discovered a **huge  mistake**  in one of the Ensembling models ( in CatBoost used ``model.predict()`` instead of ``model.predict_proba()``) that affected the whole score and our rank as seen in the table and image below 


Model  |   Public Score  |Private Score|
| --- | --- |
| [Tabnet] (https://www.kaggle.com/code/medali1992/aug-tps-tabnetclassifier) |  0.58908|0.59098 |
| [Logistic Regression](https://www.kaggle.com/code/medali1992/tps-aug-logistic-regression) | 0.58974 | 0.59018 |
| [LightGBM] (https://www.kaggle.com/code/medali1992/tps-aug-lightgbm) | 0.58381 | 0.58831 |
| [Neural network] (https://www.kaggle.com/code/nourhadrich/tps-aug-neural-network) | 0.58828 |  0.59064|
| [XGBoost](https://www.kaggle.com/code/nourhadrich/tps-aug-xgboost)| 0.58288 |  0.58898|   
| [CATBoost] (https://www.kaggle.com/code/nourhadrich/tps-aug-catboost)  | 0.58484  | 0.58964 |
| [KNN] (https://www.kaggle.com/code/nourhadrich/tps-aug-knn)|  0.58121| 0.58995 |
| [Ensembling] (https://www.kaggle.com/code/nourhadrich/tps-aug-ensembling)  | 0.58675 |  0.59090|
| [Ensembling After] (https://www.kaggle.com/code/nourhadrich/tps-aug-ensembling)| 0.58741|  0.59110|






