# 2nd Place Solution

Competition: playground-series-s3e7
Rank: #2
Source: https://www.kaggle.com/c/playground-series-s3e7/discussion/390956

Hello all,

This is the description on how I achieved the second place on this competition:

### CV
My CV was done using booking date as the feature to stratify, I decided to use this column instead of target because there were month with few observations and all of them were assigned to the same fold.

Some models I ensembled used 30 Folds and other 20 Folds just because computation was expensive. Original dataset was concated to each fold, i.e for a 20 Fold CV each fold had the entire original dataset + 95% of the train dataset.

### The Imputation
Strange dates were imputed using last day of month.

### The Duplicates
All duplicates came in pairs and **always** there was 1 positive label and one negative label. There were two leaks, most people figured out the first but the second was not disclosed. 

1. For duplicates on test that were in train, the trick was to assign the unknown label to the opposite of the known label.
2. For duplicates on test that were also in test, the trick is assign a score that maximizes oof. Explanation: if you have a pair of duplicates, any model will predict the same value for them, so any value close to 0 or 1 will hurt your model. Since AUC is a rank metric, and the positive is around 38%, then ideally, the best threshold should be in the  62nd  percentile of your predictions, this value can be used to overwrite the duplicates.

### Features
I found that most of the out of box features worked well. Some additional features I used was `is_generated` and `is_inconsistent`, both features are indicators. Dayofyear features also was selected in some models when doing Sequential Feature Selection.

### The ensemble
Final Ensemble had 16 models that were selected using Hill Climb Technique and the weights were optimized using Nelder-mead algorithm. I did not rank predictions. Most of the models were LGBM, a couple of XGBoost. Everything was trained using EarlyStopping and the test predictions were calculating using the mean of each fold-model.

### What didn't work
Holidays features, day of month feature, an indicator column that is positive whether the day was inconsistent, more than 30 Folds.

My CV was perfectly correlated with the leaderboard, I felt like if I had more time maybe I would had used some public notebooks to do a final ensemble.
