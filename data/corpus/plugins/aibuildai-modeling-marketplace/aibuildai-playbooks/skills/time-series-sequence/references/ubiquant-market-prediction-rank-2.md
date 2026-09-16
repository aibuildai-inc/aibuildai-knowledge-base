# [2nd Place Solution] - Robust CV and LGBM

Competition: ubiquant-market-prediction
Rank: #2
Source: https://www.kaggle.com/c/ubiquant-market-prediction/discussion/338615

Thanks to Ubiquant for hosting this competition!


This competition was really challenging and push me through lot's of computing optimization.


**My journey was:**
| Rank | Score | Update |
| --- | --- |
|34 | 0.082800 | first update|
|12 | 0.115900 | second update|
|4 | 0.133100 | third update|
|2 | 0.128200 | forth update|
|2 | 0.123175 | fifth update|


My model is quite simple because i work mainly on making the code robust to bugs and over-fitting.
I used every available data: train.csv + supplemental_train.csv (it was difficult to optimize the pipeline without going in out of memory)

**FE**
300 basic columns.
100 new columns: average by time id for the most correlated feature with the target on the latest 1000 time_id with more than 31 observation (statistical magic number :D)
5 macro aggregation: for each row (time_id, investment_id) i calculated mean, std, quantile 0.1, quantile 0.5, quantile 0.9 over every numerical feature (f_0, ... f_300)

**Loss/Metrics:**
rmse and correlation (correlates well with competition metrics)

I used a Purged K-FOLD cross validation with embargo so i don't have leakege between Fold, this helps to reduce over fitting. 

I train 5 LightGBM with early stopping based on CV Correlation (not on single validation score).

**What didn't work:**
AE MLP (https://www.kaggle.com/competitions/jane-street-market-prediction/discussion/224348)
Feature neutralization
PCA
