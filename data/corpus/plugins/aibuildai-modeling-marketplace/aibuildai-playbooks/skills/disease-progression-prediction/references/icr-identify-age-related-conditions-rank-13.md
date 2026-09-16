# 13th Place Solution for the "ICR - Identifying Age-Related Conditions" Competition

Competition: icr-identify-age-related-conditions
Rank: #13
Source: https://www.kaggle.com/c/icr-identify-age-related-conditions/discussion/431312

As per your comments, it was anticipated that this competition would be a shake-up, but honestly, I am surprised by this result. Just to make sure, I will provide my solution below.

# Context
Business context: https://www.kaggle.com/competitions/icr-identify-age-related-conditions/overview
Data context: https://www.kaggle.com/competitions/icr-identify-age-related-conditions/data

# Overview of the Approach section
## Data processing
・Missing Values: Create missing value flags and fill with the mean
・Column "EJ": label encoding
・Addition of Group Features: For variables other than the target, add ['min', 'max', 'mean', 'std'] of each feature grouped by the "EJ" column.

## Training
Based on the discussion by Chris below, I created three models with downsampling and added class weights during training.Validation was performed using StratifiedKFold with a value of n_splits=10 for all models, and then the results were aggregated using seed averaging.

・Model1：LGBMClassifie（features=all）
・Model2：LGBMClassifier（training with the top 20 impactful features of Model 1）
・Model3：CatBoostClassifier(features=all)

# Details of the submission
・submission["class_1"]=Model1*0.2 + Model2*0.2 + Modell3*0.6
・result：Public=0.22, Private=0.36

# Sources
・https://www.kaggle.com/competitions/icr-identify-age-related-conditions/discussion/412507
