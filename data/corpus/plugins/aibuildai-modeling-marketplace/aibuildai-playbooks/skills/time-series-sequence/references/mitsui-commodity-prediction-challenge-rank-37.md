# 37th Place Solution for the MITSUI&CO. Commodity Prediction Challenge

Competition: mitsui-commodity-prediction-challenge
Rank: #37
Source: https://www.kaggle.com/c/mitsui-commodity-prediction-challenge/writeups/andnov-37th-place-solution-for-MITSUI-CO-commodity

Thank you for Kaggle and MITSUI&CO. for organizing this competition. My best solution was actually the simple one.


## **Context**

- Business context: https://www.kaggle.com/competitions/mitsui-commodity-prediction-challenge

- Data context: https://www.kaggle.com/competitions/mitsui-commodity-prediction-challenge/data


## **Model**

Multiregression with CatBoostRegressor

`params = {
            'loss_function' : 'MultiRMSEWithMissingValues',
            'eval_metric' : 'MultiRMSEWithMissingValues',
            'iterations' : 100,
            'max_depth' : 5,
            'learning_rate' : 0.05
            }`


## **Features**

No Features Engineering, only use all target features and volume features


## **Cross Validation**

Using KFold with 5 fold and submission made of averaging the prediction from each model saved in the fold


## **Things that is not working / improving my model**

- Use percentage change of the features instead of actual features.
- Adding more iterations
- Use NN with Pytorch MLP,  Embedding, CNN
