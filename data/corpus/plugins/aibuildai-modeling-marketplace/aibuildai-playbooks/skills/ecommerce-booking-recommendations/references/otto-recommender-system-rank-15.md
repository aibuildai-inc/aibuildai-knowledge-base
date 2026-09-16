# 15th Place Solution

Competition: otto-recommender-system
Rank: #15
Source: https://www.kaggle.com/c/otto-recommender-system/discussion/382905

First of all, I would thank the organizers and those who shared their knowledge and ideas. Especially, I would like to thank @carnozhao for providing great numba notebook. Without him, I must give up this competition and enjoy  my New year holiday lying in the bed everyday (LOL.)

## Approch Summary
- Retrieval & Re-rank 2stage model
- local cv: train on week3/ valid on week 4
- Retrieval recall20: 0.585(pulic LB)/0.585(private LB)
- Re-rank models: lightgbm ranker with 182 features. (single model get 0.601(public LB)/0.600(private LB))
- ensemble: 5 lightgbm ranker with same features with different seed  for negative sampling and model training. ( 0.601(public LB)/0.601(private LB))
## Retrieval/candidate stage
- 100 candidates for each session. (candidates number 50 -> 100, which boosts 0.0007. have no time and memory to test 200.)
    - historical aids.
    - i2i2i based on @carnozhao 's notebook 
    - most popular aids to make up for 100 candidates.
- local cv on week4 [LB 0.585]

| num of candidates | click |cart|order|sum|
| --- | --- |--- |--- |--- |
| 20 | 0.5397 |0.4220|0.6580|0.5754|
|50|0.6223|0.4838|0.69498|0.6243|
|100|0.6745|0.5265|0.7188|0.6567|

- main modification of i2i
    - order weight: adjacent aids in interactions have larger weight. (session 1 has a,b,c aids, the weight of b,c is 1 and the weight of a,c is 1/2.) Use those weight in both i2i similarity and u2i recommandation. I didn't use time weight here, because I notice most of interactions are within 1hour.
    -  Normalize the i2i similarity dict. The popular aids have larger i2i score with other aids. It's not fair in the u2i stage. Just use the mean of i2i scores of popular aids to normalize.

## Rerank-models
lightgbm ranker with 182 features. 0.2 negative ratio. (single model get 0.601(public LB)/0.600(private LB))

- session features(adding more session features didn't work for me)
    - event count/ type count
    - aid_nunique/ type aid_nunique
    - first time, last time, time range
- aids features 
    - kinds of count/nunique from (1week,2week)
    - hour features to detect some "promotion" aids which only exists for 1-2 hour in a week.
    - ratio of buy2cart, buy2order, cart2order
    - ratio of re-cart, re-order, re-click
    - first last time of aids 
- aid X session features
    - kinds of count, order decayed counts
    - time diff
- retrieval features
    - revisit rank/scores
    - i2i rank/scores
    - (avg/median/max/last one / last two/ last 3 avg) i2i scores
- embedding similarity
    - (avg/median/max/last one / last two/ last 3 avg) w2v similarity
    - (avg/median/max/last one / last two/ last 3 avg) bpr similarity
## ensemble
- 5 lightgbm ranker with same features with different seed  for negative sampling and model training. 
| LB|  click|cart|order|sum|
| --- | --- |--- |--- |--- |
| public |  0.056|0.136|0.408|0.601|
| private|  0.056|0.136|0.408|0.601|

- catboost&&MLP ensemble can boost 0.0003 in local cv but have no time to run it.
## what didn't work for me.
- prone similarity
- click/cart/order 3 targets are correlated. So I use pred score of other 2 targets as features.(i.e. when predicting order, I use pred score of click and cart). In local cv, it boosted almost 0.001 but didn't work in LB. don't know why.
- combining week3 data and week4 data as final trainset didn't work. I don't create week2 data to test it in local cv. I just used 1.25 * num of rounds of lgb using only week4 to train.
- adding more session feas and interaction feas.
- adding k nearest neighbor of w2v embedding in the retrieval stage didn't improve hit rate.
