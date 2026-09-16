# 6th Position winner solution for the : ICR - Identifying Age-Related Conditions

Competition: icr-identify-age-related-conditions
Rank: #6
Source: https://www.kaggle.com/c/icr-identify-age-related-conditions/discussion/431048

My solution is divided into 7 big steps:
(Code version 10)
1. Interpolate the missing data using a linear method using the 'interpolate' instance from pandas.
2. Use a random forest classifier to find the most important features in the dataset using 'gini-importance'.
3. Use Bayesian optimization to find the optimal parameters of the XGBoost classifier.
4. Repeat step 3 multiple times to gather many optimal parameters for the XGBoost classifier.
5. Make an ensemble of XGBoost classifiers using the optimal parameters.
6. Fine-tune the XGBoost classifiers again using GridSearchCV (because Bayesian optimization is just an estimation of the parameters).
7. Use a voting classifier (the mean of the probabilities of each XGBoost) to classify the test set.

Here is my code (I performed step 2 in my personal computer):
https://www.kaggle.com/code/diegosilvadefrana/notebooke87ef51e7e/notebook
