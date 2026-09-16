# 9th Place Solution XGB stack

Competition: playground-series-s3e7
Rank: #9
Source: https://www.kaggle.com/c/playground-series-s3e7/discussion/390961

Congratulations to the winners of this competition! It was a head-to-head run for top 10 contenders except the first place. 

My solution:

**CV**:
30 Fold stratified. + Inner 5-Fold.

**RFE**: 
I have removed 2 features `no_of_previous_cancellations`, `no_of_previous_bookings_not_canceled`. (*I burned quite a lots of cores to find a good combination out of 800+*).

**Date Anomalies** 
were fixed by imputing the max date of the month by following [the discussion I initiated in the very beginning] (https://www.kaggle.com/competitions/playground-series-s3e7/discussion/386655).

**Target Encoding**:
All categorical features were target encoded by using smooth mean algorithm with 2 level CV to prevent overfitting.

**Target transformation**:
I have transformed some of the contradicting duplicated targets with mode. (I have assigned the most common value as the target to the duplicated rows. 50%-50% examples I have left untouched). 

**Ensemble**:
Hill Climbing. 7 Ensembled XGB models.

**Postprocessing (Date Leak Boost)**:
was implemented in accordance with [Data Leak Code Snippet](https://www.kaggle.com/competitions/playground-series-s3e7/discussion/388992).

**What did not work**:
Honestly, everything worked. Every idea I tried improved my CV and LB until the very last moment. At some point I got stuck with a bunch of well trained XGB models. I have discarded LGBM and CatBoost since the yielded worst CV - LB score. Also their score correlation with LB was awful.

**What could be improved**:
I took an immense task to write double cross-validation frameworks for XGB, LGBM, CatBoost which took too much time and efforts. In the end of the day I made it work and the models with the target encoding stopped to overfit. I spent too much time creating smart features, most of which proved not working. Although one of my model with smart features yielded straight Top 10 (**mb I don't know something and it is better to throw everything into the XGB-LGBM-CAT auto-solver**).  I felt as I over-engineered the things themselves. Top 10 is a reasonable achievement for still not polished frameworks.

**Update, after applying 0.5 to 253 pairs of duplicates in the test was giving straight top1 public and top2 private with a margin, at least my ensemble was really good!**




**On top of the above mentioned, here is what should have been done:**
```python
test = pd.read_csv('data/test.csv')
idx = test[test.drop(columns='id').duplicated(keep=False)].id.values
sub = pd.read_csv('submissions/hill_20_19_17_21_5_6_11__wmean_public_leak.csv')
sub.loc[sub.id.isin(idx), 'booking_status'] = 0.5
sub.to_csv('submissions/0.5.csv', index=False)
```
