# 13th Place Solution

Competition: playground-series-s3e5
Rank: #13
Source: https://www.kaggle.com/c/playground-series-s3e5/discussion/386657

Hi! Congratulations to everyone on their results, and huge kudos to my teammate @karakasatarik! It was another playground competition that made us learn a new metric and concept again. I would like to briefly talk about our solution.

#### Approaches:

- Symbolic Regression [(Public Approach)](https://www.kaggle.com/code/paddykb/ps-s3e5-repeated-sr-with-inequalities)
- Multi-Class Neural Networks [(Modified Version of this Public Approach)](https://www.kaggle.com/code/paddykb/ps-s3e5-keras-weighted-kappa-loss)
- XGB/LGB Cherrypick Ensemble [(Public Approach)](https://www.kaggle.com/code/usedpython/s3e5-0-624-xgb-lgb-kmeans-pca)
- CatBoost Nested CV Blending (Private Approach)


#### Features Used

```
df['mso2'] = df['free sulfur dioxide']/(1+ 10**(df['pH'] -1.81))

df['acidity_ratio'] = df['fixed acidity'] / df['volatile acidity']
df['total_acid'] = df['fixed acidity'] + df['volatile acidity'] + df['citric acid']
df['mean_acid'] = df[['fixed acidity','volatile acidity','citric acid']].mean(axis=1)
df['std_acid'] =  df[['fixed acidity','volatile acidity','citric acid']].std(axis=1)

df['free_sulfur/total_sulfur'] = df['free sulfur dioxide'] / df['total sulfur dioxide']
df['sugar/alcohol'] = df['residual sugar'] / df['alcohol']

df['sugar/citric'] = df['residual sugar'] / df['citric acid']

df['BSO2'] = df['total sulfur dioxide'] - df['free sulfur dioxide']
df['FSO2/alcohol'] = df['free sulfur dioxide'] / df['alcohol']
df['TSO2/alcohol'] = df['total sulfur dioxide'] / df['alcohol']
df['BSO2/alcohol'] = df['BSO2'] / df['alcohol']

df['chlorides/TSO2'] = df['chlorides'] / df['total sulfur dioxide']
df['sulphates/pH'] = df['sulphates'] / df['pH']

df['alcohol/density'] = df['alcohol'] / df['density']
df['alcohol_density'] = df['alcohol']  * df['density']
df['sulphates/chlorides'] = df['sulphates'] / df['chlorides']
df['alcohol/pH'] = df['alcohol'] / df['pH']
df['alcohol/acidity'] = df['alcohol'] / df['total_acid']
df['alkalinity'] = df['pH'] + df['alcohol']
df['mineral'] = df['chlorides'] + df['sulphates'] + df['residual sugar']
df['density/pH'] = df['density'] / df['pH']
df['total_alcohol'] = df['alcohol'] + df['residual sugar']

df['acid/density'] = df['total_acid']  / df['density']
df['sulphate/density'] = df['sulphates']  / df['density']
df['sulphates/acid'] = df['sulphates'] / df['volatile acidity']
df['sulphates*alcohol'] = df['sulphates'] * df['alcohol']
```


#### Submissions
| Method | Public Score | Private Score | Selected |
| --- | --- |
| Symbolic Regression | 0.62977 | 0.57276 |
| Multi-Class Neural Networks (Modified) | 0.60113 | 0.58702 |
| XGB/LGB Cherrypick Ensemble | 0.62485 | 0.58220 |
| CatBoost Nested CV | 0.59901 | 0.57403 |
| 8 CatBoost Nested CV + 1 NN | 0.59030 | 0.57623 | X |
| SR + Modified NN + XGB/LGB Cherrypick | 0.62949 | 0.59121 | X |
| **(Best)** SR + CatBoost Nested CV + XGB/LGB Cherrypick | 0.61127 | 0.59656 |

It was a fun competition, it reminded us to trust CV and diversity in blending again!
