# Congratulations!

Competition: avito-context-ad-clicks
Rank: #5
Source: https://www.kaggle.com/c/avito-context-ad-clicks/discussion/15606#87421

Congratulations to the winners!!! That was a unique big data competition!

Our aproach is based in two algos: FTRL and XGBoost. Basically we build 3 models  for each algo, then found some weights to combine then all.

FTRL was built over all dataset:
Model1 -sorted randomly.  Public LB: 0.04277
Model2 -sorted by UserID then Date.  Public LB: 0.0425x
Model3 -sorted by AdID then Date.  Public LB: 0.0425x
Three models combined: 0.4235

XGBoost was build over last 8 clicks of each UserID:
Model 1, 2 and 3 were built using different features and hyperparameters, including all datasets, some time based features like counts per hour, history of clicks by User, by Ad. Also we created 6 xgb meta features using the dataset composed from clicks first to last 9 of each User to train. Also for features with too many levels like UserID and AdID we aplied a smoothing and also combined 2-way with other features to generate more featuers. Our XGB model 1 nd 2  have aprox. 50 feats and 3 about 100 feats.
Our best XGB model 2 scored Public LB: 0.04108

Golden Features: before SearchStream.tsv filter by ObjectType==3:   Sum of Objective==1 by SearchID, Sum of Objective==2 by SearchID, number of instances by SearchID.   Combine these 3 features.

We trained and validated using the last click of each UserID. 
Combining XGB and FTRL scored 0.04088 public and private 0.04107.
Working with that dataset using a low end hardware is very hard :-( so we worked hard only in the last 7 days of the competition.
