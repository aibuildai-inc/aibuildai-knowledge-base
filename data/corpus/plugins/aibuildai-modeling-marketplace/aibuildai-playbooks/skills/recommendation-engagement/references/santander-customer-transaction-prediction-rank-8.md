# #8 team solution

Competition: santander-customer-transaction-prediction
Rank: #8
Source: https://www.kaggle.com/c/santander-customer-transaction-prediction/discussion/88886#latest-512882

A short summary of our (and probably others) magic. Its about counts. Counts cannot be captured by lgbm on raw data.
With 400feats: [raw,isNotUnique*raw] + lgbmUpsampled we got 0.924 on public.
"isNotUnique" filters out samples with unique numeric value (cnt==1, calculated on 300k rows [train,testReal]).

Then we added a few variations of lgbm/xgb models and linear blended them - thats our 0.925 public score.
Variations on data are (just minor improvements)
- added counts (+200f)
- filtered out values which appear 1 or 2 times (+200f)
- leaky features which calc isNotUnique*raw on target==1/target==0 train samples (+400f)

We tried tons of tricks, almost all failed. A few mentioned: other features, dae pretraining, nn, catboost, more upsampling, pseudo labeling, feature selection, target encoding, parameter tweaking.
The most weird thing is the "shallow"/"independent" property of the dataset, on our 400f I got 0.921 with naiveBayes. It would be interesting to get more infos about santander's anonymization.
Lets see what top teams did.

Michael
