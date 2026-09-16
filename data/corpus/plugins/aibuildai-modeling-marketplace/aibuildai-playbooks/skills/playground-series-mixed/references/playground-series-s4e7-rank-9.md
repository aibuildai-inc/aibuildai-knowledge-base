# #9 Solution | 24 Models + Hill Climbing

Competition: playground-series-s4e7
Rank: #9
Source: https://www.kaggle.com/c/playground-series-s4e7/discussion/523403

First, I would like to express my gratitude to the Kaggle team for organizing this competition. It was my first time handling such a large dataset, which made the project exciting from the start. In this post, I will provide a brief overview of my approach. [](url)

# Initial Modeling 
In the initial days of the competition, due to the large size of the data, I chose to model the data using `Logistic Regression`. This approach provided some initial insights about the data. For example, the performance of the `Logistic Regression` model improved significantly when I used `TargetEncoding`. This suggests that certain features, even though stored as numbers, should be considered categorical. You can find more details in this [post](https://www.kaggle.com/competitions/playground-series-s4e7/discussion/516462).

# Data
I used both the competition and original datasets.

# Data preprocessing
In order to reduce memory usage, I considered the following preprocessing steps:

```
def converting_datatypes(df, df_train=False):
    
    df = df.copy()
    
    if df_train==False:
        
        df['Policy_Sales_Channel'] = np.where(df['Policy_Sales_Channel']==144., 145., df['Policy_Sales_Channel'])
        df['Policy_Sales_Channel'] = np.where(df['Policy_Sales_Channel']==149., 150., df['Policy_Sales_Channel'])

    df['Age'] = df['Age'].astype('int8')
    df['Driving_License'] = df['Driving_License'].astype('int8')
    df['Region_Code'] = df['Region_Code'].astype('int8')
    df['Previously_Insured'] = df['Previously_Insured'].astype('int8')
    df['Annual_Premium'] = df['Annual_Premium'].astype('int32')
    df['Policy_Sales_Channel'] = df['Policy_Sales_Channel'].astype('int16')
    df['Vintage'] = df['Vintage'].astype('int16')
    df['Gender'] = df['Gender'].map({'Female': 0, 'Male': 1}).astype('int8')

    df['Vehicle_Age'] = df['Vehicle_Age'].map({'< 1 Year': 0, 
                                               '1-2 Year': 1,
                                               '> 2 Years': 2}).astype('int8')

    df['Vehicle_Damage'] = df['Vehicle_Damage'].map({'No': 0, 'Yes': 1}).astype('int8')

    if df_train==True:

        df['Response'] = df['Response'].astype('int8')

    return df
```

# Feature Engineering
I considered the following features:

```
def fe(df_train, df_test, df_original):

    n = df_train.shape[0]
    m = df_test.shape[0]
    p = n+m
    df_tot = pd.concat([df_train, df_test, df_original], axis=0).reset_index(drop=True)
    
    df_tot['interaction_1'] = pd.factorize((df_tot['Previously_Insured'] + df_tot['Vehicle_Age']).to_numpy())[0]
    df_tot['interaction_2'] = pd.factorize((df_tot['Previously_Insured'] + df_tot['Vehicle_Damage']).to_numpy())[0]
    df_tot['interaction_3'] = pd.factorize((df_tot['Previously_Insured'] + df_tot['Vintage']).to_numpy())[0]
    df_tot['interaction_4'] = pd.factorize((df_tot['Previously_Insured'] + df_tot['Annual_Premium']).to_numpy())[0]
    df_tot['interaction_5'] = pd.factorize((df_tot['Previously_Insured'] + df_tot['Gender']).to_numpy())[0]
    df_tot['interaction_6'] = pd.factorize((df_tot['Previously_Insured'] + df_tot['Driving_License']).to_numpy())[0]
    df_tot['interaction_7'] = pd.factorize((df_tot['Vehicle_Age'] + df_tot['Vehicle_Damage']).to_numpy())[0]
    df_tot['interaction_8'] = pd.factorize((df_tot['Vehicle_Age'] + df_tot['Driving_License']).to_numpy())[0]


    return [df_tot[:n], df_tot.iloc[n:p].drop(columns=['Response'], axis=1), df_tot[p:]]
```

# Models

I considered the following models:

* Random Forest (under the LGBM framework because it runs faster). For more details, see this [pos](https://www.kaggle.com/competitions/playground-series-s4e7/discussion/520529#2924787).
* LGBMClassifier
* XGBClassifer
* TensorFlow. The first version of the TensorFlow I built reached a 0.882 oof ROC-AUC over 10 folds. Then, I used @paddykb TensorFlow model (see this [notebook](https://www.kaggle.com/code/paddykb/ps-s4e7-keras-haz-insurance-losses)) because it outperformed my initial TensorFlow model.  
* Catboost. I think everyone in this competition has used some sort of version of the @rohanrao Catboost model. For more detail see this [notebook](https://www.kaggle.com/code/rohanrao/automl-grand-prix-1st-place-solution).

The following table show the best performance over 10-folds of each of the considered models.

| Model | Competition Data | Competition + Original Data |
| --- | --- | --- |
| CatBoost | 0.895311 | 0.895819 |
| LGBM | 0.892605 | 0.892753 |
| TensorFlow | 0.892083 | 0.892213 |
| XGBoost | 0.890959 | 0.89105 |
| Random Forest | 0.873091 | 0.875128 |

# Ensemble
In the Kaggle community, it is widely acknowledged that the `ROC-AUC` score can be optimized by the ensemble of different model predictions. To achieve this, I employed the hill-climbing strategy. This strategy, akin to climbing a hill, involves combining model predictions in a linear fashion, always moving upwards as long as the objective metric improves. The final ensemble includes 24 models: 

| Model | Competition Data | Competition + Original Data |
| --- | --- | --- |
| CatBoost | 4 models | 4 models |
| LGBM | 3 models | 3 models |
| TensorFlow | 3 models | 3 models |
| XGBosst | 1 model | 1 model |
| Random Forest | 1 model | 1 model |

From the above table, it is important to note that not the same models were trained in the data. For instance, 4 `CatBoost` models were trained on the competition data; however, different features were used to trained those models.

# Post Processing
Finally, I post-process the predictions of the hill-climbing ensemble using @paddykb suggestion presented in this [post](https://www.kaggle.com/competitions/playground-series-s4e7/discussion/520253).

I am looking forward to the next episode.
