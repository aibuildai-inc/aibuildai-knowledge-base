# 47th Place Solution | RepeatedKFold CatBoost

Competition: playground-series-s3e17
Rank: #47
Source: https://www.kaggle.com/c/playground-series-s3e17/discussion/419708

This was my first competition, so I'm glad it went well. The solution uses the CatBoostClassifier algorithm with repeated K-Fold cross validation to split the data into training and validation sets. 

My notebook [link to the notebook](https://www.kaggle.com/annafabris/repeatedkfold-catboost)

## Strategies adopted
- Utilizing both the training data and the original dataset
- Feature selection: I removed just the "id" and "UID" columns
- CatBoostClassifier: The decision to use the CatBoostClassifier algorithm yielded highly satisfactory results. Its built-in handling of categorical features proved to be a winning choice.
- Repeated K-Fold cross validation: I adopted repeated k-fold cross-validation to create robust training and validation sets

I would like to acknowledge the following valuable resource that contributed to my understanding and implementation of CatBoostClassifier in this competition:

CatBoost notebook by Jimmy Yeung: [link to the notebook](https://www.kaggle.com/code/jimmyyeung/ps3-17-machine-failure-catboost-top-21)
