# 6th place solution

Competition: playground-series-s4e7
Rank: #6
Source: https://www.kaggle.com/c/playground-series-s4e7/discussion/523484

Notebook: https://www.kaggle.com/code/ravaghi/insurance-cross-selling-6th-place-solution

## Data Preprocessing

I used the original dataset in addition to the competition dataset to train all my models. I changed the data types to reduce memory usage, converted categorical features to numerical values using simple mappings, and added the following features which I borrowed from [this notebook](https://www.kaggle.com/code/rohanrao/automl-grand-prix-1st-place-solution).

```python
dataframe['Previously_Insured_Annual_Premium'] = pd.factorize(dataframe['Previously_Insured'].astype(str) + dataframe['Annual_Premium'].astype(str))[0]
dataframe['Previously_Insured_Vehicle_Age'] = pd.factorize(dataframe['Previously_Insured'].astype(str) + dataframe['Vehicle_Age'].astype(str))[0]
dataframe['Previously_Insured_Vehicle_Damage'] = pd.factorize(dataframe['Previously_Insured'].astype(str) + dataframe['Vehicle_Damage'].astype(str))[0]
dataframe['Previously_Insured_Vintage'] = pd.factorize(dataframe['Previously_Insured'].astype(str) + dataframe['Vintage'].astype(str))[0]
```
I was initially hesitant to use these features due to their leaky nature, but they improved both my CV and public LB scores, so I decided to keep them.


## Models

The following models were used in my ensemble:
- CatBoost ([notebook](https://www.kaggle.com/code/ravaghi/s04e07-insurance-cross-selling-catboost))
- LightGBM ([notebook](https://www.kaggle.com/code/ravaghi/s04e07-insurance-cross-selling-lightgbm))
- XGBoost ([notebook](https://www.kaggle.com/code/ravaghi/s04e07-insurance-cross-selling-xgboost))
- Neural Network ([notebook](https://www.kaggle.com/code/ravaghi/s04e07-insurance-cross-selling-ann))
- Logistic Regression

I saved the OOF predictions and test predictions of each model and used the resulting files in my ensemble. Note that the provided notebooks don't have the exact set of hyperparameters that I used, but the rest of the code is the same as what I used for my final models.

For CatBoost and Logistic Regression, I treated all features as categorical. For the neural network, I one-hot encoded some of the features and target encoded one of the features as suggested [here](https://www.kaggle.com/competitions/playground-series-s4e7/discussion/518760).


## Ensemble

I used a simple `StackingClassifier` as my ensemble technique. To save time and prevent `StackingClassifier` from training every model from scratch, I used [this trick](https://www.kaggle.com/competitions/playground-series-s4e6/discussion/509353#2851035) that I learned in the last competition. I log-transformed the OOF predictions of my base models and fed them through the model.

By default, `StackingClassifier` uses `LogisticRegression` as its final estimator. I tried tuning `LogisticRegression`, but it didn't help. I also tried a tuned `XGBClassifier` and `LGBMClassifier`, but they didn't help either, so I decided to stick with the default settings.

## Post Processing

I applied *[the glitch in the insurance matrix trick](https://www.kaggle.com/competitions/playground-series-s4e7/discussion/520253)* by @paddykb to my test predictions and improved my score by ~0.0006, which is the same improvement as @paddykb reported.

## Results

Here is the CV scores of each of my models trained on 5 folds.

