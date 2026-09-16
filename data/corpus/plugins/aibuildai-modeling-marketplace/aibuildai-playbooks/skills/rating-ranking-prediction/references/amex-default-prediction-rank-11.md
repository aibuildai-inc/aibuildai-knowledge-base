# 11th Place Solution (LightGBM with meta features)

Competition: amex-default-prediction
Rank: #11
Source: https://www.kaggle.com/c/amex-default-prediction/discussion/347786

※Previously titled 12th Place Solution. It is now 11th due to the fixed ranking.

Thank you to everyone who participated in the competition and to everyone involved in organizing it.
I learned a lot through this competition.

## Score & Result
- My best submission
    - Local CV：0.79922
    - Public: 0.80088
    - Private: 0.80852
- Results
    - Public:6th → Private 12th

## Feature Engineering
- Base features are from public notebook
    - https://www.kaggle.com/code/thedevastator/amex-features-the-best-of-both-worlds
- Delete some features
    - Round up features
    - Duplicate features (ex. groupby counts for all category features)
- Add some features
    - Features aggregated by time period 
        - min, max, mean, std for the last 3 and last 6 months
    - Rate and diff features in time series
        - ex. last - latest_3month_mean, last_3month_mean / last_6month_mean
    - Null count features
    - Date features
    - **Meta features (most important features!)**
        - how to make
            1. Train_labels are assigned to train data (before aggregation by cid) and train model.
            2. Make oof prediction for train data.
            3. Aggregate oof prediction by time period.
        - Using this feature, I reached 0.800 PublicLB from 0.799 PublicLB in single model.
        - Referring to the DSB2019's 2nd place solution method.
            - https://www.kaggle.com/c/data-science-bowl-2019/discussion/127388

## Validation strategy
- Use stratfiedKfold(k=5).
- I think Public LB is more important than local CV to measure Private's performance.
    - Data size is about the same for train and public.
    - In terms of time, public data is closer to private data than train data.
    - Even after adversarial validation, train/private was farther away from the data than public/private.
        - train/private：AUC 0.99
        - public/private：AUC 0.82
- While focusing on publicLB, we also looked at local CV to determine if there was any improvement.
    - Also checked local logloss because amex_metric was not stable.
    - It was hard to find a few digits of publicLB.

## Model
- LightGBM
    - Use dart.
    - Hypyer_parameter is the same as base notebook.
        - https://www.kaggle.com/code/thedevastator/amex-features-the-best-of-both-worlds
    - Get best amex metric model (use callback)
        - https://www.kaggle.com/competitions/amex-default-prediction/discussion/332575    

## Feature Selection
- Adversarial validation
    - Delete features of high importance in train/private adversarial validation.
        - Drop_features: R_1, D59, S_11, B_29
        - After the change, the AUC was 0.8.
- Null importance
    - Use features actual importance larger than mean importance with shuffled target.
    - Before:4300 features → After:1300 features

## Ensemble
- Use 3 LightGBM models and rank ensemble (weighted average).
- Each model use different feature set.
    - Model_1: Not use meta features.
    - Model_2: Use meta features & large features (not use null importance feature selection).
    - Model_3: Use meta features & small features (use null importance feature selection).
- Ensemble weight
    - model_1:model_2:model_3 = 4:4:2
    - weight decided while looking at public LB

## Select Submission
- I chose two sub's: 
    1. BestLB sub
    2. Sub with risk of time-series changes in features.
        - Features that were not important in adversarial validation are not used. 
- The second model was the best in privateLB.
- Perhaps the trend of some features changed over time. adversarial validation was very helpful.


Thank you.
