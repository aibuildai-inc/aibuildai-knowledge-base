# 4th place solution: Stacking with XGB + Pseudo labeling + metric optimizing.

Competition: playground-series-s4e2
Rank: #4
Source: https://www.kaggle.com/c/playground-series-s4e2/discussion/480939

Thanks again for an interesting series challenge, always a fun and good area for testing and experiment new models and ideas, benchmark against the old ones!

Big shakeup in this competition, personally I moved 255 places. But this is expected with a 20/80 ratio in the test set, and one should heavily weight the best local cross-validation in such scenario.

In this challenge we had a multi-class problem, we had similar not so long ago in the Multi-Class Prediction of Cirrhosis Outcomes so I picked up some ideas from there from my 4th place solution. 
https://www.kaggle.com/competitions/playground-series-s3e26/discussion/464863
Alright it’s not the same, different features, metric, data etc. but still the concept can be reused. And what a positive result, I landed in 4th place here as well.

**Solution**

**Summary**

A stacking approach with XGB as meta learning and different SOTA solutions as extra added stacking features from each prediction. Final inference includes optimized accuracy metric and 9 predictions of different version of the stacking code and other diverse solutions for a count of max class per row.

**Data and Feature Engineering**

Both the competition dataset and the extra dataset was used.
I used two different FE for training:

```python
train_df = pd.read_csv('/kaggle/input/playground-series-s4e2/train.csv')
original = pd.read_csv('/kaggle/input/obesity-or-cvd-risk-classifyregressorcluster/ObesityDataSet.csv')
train_df = pd.concat([train_df, original]).drop(['id'], axis=1).drop_duplicates().reset_index(drop=True)

# Advanced feature engineering for training data
train_df['Age_Group'] = pd.cut(train_df['Age'], bins=[20, 30, 40, 50, 55], labels=['A', 'B', 'C', 'D'],)
train_df['Log_Age'] = np.log1p(train_df['Age'])
scaler = MinMaxScaler()
train_df['Scaled_Age'] = scaler.fit_transform(train_df['Age'].values.reshape(-1, 1))
#train_df = train_df.drop(columns=['id'])

test_df = pd.read_csv('/kaggle/input/playground-series-s4e2/test.csv')

# Advanced feature engineering for test data
test_df['Age_Group'] = pd.cut(test_df['Age'], bins=[20, 30, 40, 50, 55], labels=['A', 'B', 'C', 'D'],)
test_df['Log_Age'] = np.log1p(test_df['Age'])
test_df['Scaled_Age'] = scaler.transform(test_df['Age'].values.reshape(-1, 1))

test_df = test_df.drop(columns=['id'])

label_encoder = LabelEncoder()
label_encoder.fit(train_df['NObeyesdad'].unique())
train_df['NObeyesdad'] = label_encoder.transform(train_df['NObeyesdad'])
```
And

```python
train['Age group'] = pd.cut(train['Age'], bins=[0, 18, 30, 45, 60, train['Age'].max()], labels=['0-18', '19-30', '31-45', '46-60', '60+'])
train['BMI'] = train['Weight'] / (train['Height'] ** 2)
test['Age group'] = pd.cut(test['Age'], bins=[0, 18, 30, 45, 60, test['Age'].max()], labels=['0-18', '19-30', '31-45', '46-60', '60+'])
test['BMI'] = test['Weight'] / (test['Height'] ** 2)
original['Age group'] = pd.cut(original['Age'], bins=[0, 18, 30, 45, 60, original['Age'].max()], labels=['0-18', '19-30', '31-45', '46-60', '60+'])
original['BMI'] = original['Weight'] / (original['Height'] ** 2)

train['Age * Gender'] = train['Age'] * train['Gender']
test['Age * Gender'] = test['Age'] * test['Gender']
original['Age * Gender'] = original['Age'] * original['Gender']        

categorical_features = ['Gender', 'family_history_with_overweight', 'Age group', 'FAVC','CAEC', 'SMOKE','SCC', 'CALC', 'MTRANS']
train = pd.get_dummies(train, columns=categorical_features)
test = pd.get_dummies(test, columns=categorical_features)
original = pd.get_dummies(original, columns=categorical_features)

polynomial_features = PolynomialFeatures(degree=2)
X_poly = polynomial_features.fit_transform(train[['Age', 'BMI']])
train = pd.concat([train, pd.DataFrame(X_poly, columns=['Age^2', 'Age^3', 'BMI^2', 'Age * BMI', 'Age * BMI2', 'Age * BMI3'])], axis=1)
X_poly = polynomial_features.transform(test[['Age', 'BMI']])
test = pd.concat([test, pd.DataFrame(X_poly, columns=['Age^2', 'Age^3', 'BMI^2', 'Age * BMI', 'Age * BMI2', 'Age * BMI3'])], axis=1)
X_poly = polynomial_features.transform(original[['Age', 'BMI']])
original = pd.concat([original, pd.DataFrame(X_poly, columns=['Age^2', 'Age^3', 'BMI^2', 'Age * BMI', 'Age * BMI2', 'Age * BMI3'])], axis=1)

```

**Models and frameworks used to the stacking approach.**

Competition metric is Accuracy but for training log_loss was set and probability per solution was saved for later use.

- AutoXGB
- AutoGluon with the new zero-shot HPO training.
Ensemble Weights: {'CatBoost_r9_BAG_L1': 0.363, 'LightGBM_r131_BAG_L1': 0.253, 'XGBoost_BAG_L1': 0.099, 'XGBoost_r33_BAG_L1': 0.099, 'NeuralNetTorch_BAG_L2': 0.077, 'ExtraTreesEntr_BAG_L2': 0.033, 'NeuralNetFastAI_BAG_L2': 0.022, 'CatBoost_BAG_L2': 0.022, 'NeuralNetTorch_BAG_L1': 0.011, 'RandomForestGini_BAG_L2': 0.011, 'ExtraTreesGini_BAG_L2': 0.011}

- LightAutoml with LGBM and Catboost.
- Custom XGB + LGBM training code with and without Pseudo Label training.

**Stacking**

Used the features from training and added the probability from the different solution as extra features. The meta model was XGB. Stacking is a great approach in classification problem versus other techniques.

**Metric optimizing and final inference.**

I used the public code for optimizing the metric and used earlier saved probabilities per class for it.
The public code for optimizing can be find here https://www.kaggle.com/code/samlakhmani/easy-92-196-single-model.

For the final inference I used a diverse 9 prediction collection of both own trained solutions and 2 public notebooks and calculated the max class per row.

-----------------------------------------

That’s it!!
Happy Kaggling!
