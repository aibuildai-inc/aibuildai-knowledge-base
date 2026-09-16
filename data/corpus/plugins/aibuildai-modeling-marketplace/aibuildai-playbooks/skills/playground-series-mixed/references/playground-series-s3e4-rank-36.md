# 36th place solution

Competition: playground-series-s3e4
Rank: #36
Source: https://www.kaggle.com/c/playground-series-s3e4/discussion/382493

I used the same strategy as it was before.

1) Added the original data because ROC AUC between original and test was lower than even between train and test.
2) Applied dropping for each model seperately based on Permutation Importance with corr sense.
3) Dropped the duplicates.
4) Used optuna to find the best proportion of the weights in the ensemble.

The work is here https://www.kaggle.com/code/viktortaran/ps-jan-4-2023
A good article about the real competition is here https://www.kaggle.com/competitions/ieee-fraud-detection/discussion/111284
