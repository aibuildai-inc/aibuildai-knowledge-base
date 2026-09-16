# 7th Place Solution and my first Gold and that too Solo one !

Competition: ncaam-march-mania-2021
Rank: #7
Source: https://www.kaggle.com/c/ncaam-march-mania-2021/discussion/231274

Hi All, 
First of all congrats to all folks who have participated in this competition. It was my first Solo gold in Kaggle. 
Approached used in this model
1. Feature Engineering on three data sets
a) MRegularSeasonCompactResults
b) MRegularSeasonDetailedResults
c) MMasseyOrdinals
2. Calculation of features like
a) Win%(As per all location and All)
b) Loss %(As per all location and All)
c) Score Difference(Win Score-Loss Score)
d) Calculating Relative Score on the basis of Relative Score
e)Aggregating all results on the basis of Season and Team
f) Point Difference as per MRegularSeasonDetailedResults
g) Win% and Loss % as per MRegularSeasonDetailedResults
h) last14days_stats_T1  where day >118 as per win count
i) last21days_stats_T1  
j) Statistical column on the basis of above tables
k) MOR_127_128  on MMasseyOrdinals where MMOrdinals.RankingDayNum == 127 and MMOrdinals.RankingDayNum == 128
l) MOR_50_51 as per above logic
m) MOR_15_16
n) Merging all three results above and and finding max and statistical columns out of that
2. Model used
I). LightGBM classifier 
a) GroupKFold(n_splits=4)
b) CV mean score: 0.5590, std: 0.0167.
II) XGB with tree method gpu_hist and learning rate .14
III) GroupKFold(n_splits=4)
3. Merging results as per two models

I tried to keep my approach simple.

Regards
Kamal
