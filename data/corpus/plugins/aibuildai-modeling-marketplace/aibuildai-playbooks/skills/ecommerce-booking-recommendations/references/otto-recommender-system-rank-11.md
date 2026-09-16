# 14th Place Solution(Ethan&qyxs part)

Competition: otto-recommender-system
Rank: #11
Source: https://www.kaggle.com/c/otto-recommender-system/discussion/383374

Thanks to OTTO and kaggle hosting such a good competition. Thanks my great teammates @juzqyxs @rib @ria.
## Validation ##
use the sessions in last week.

## Retrieval ##
two traditional methods,
- u2i: the aids user action before
- i2i: itemcf, we do it seperately for each task. And we used some kinds of weight to optimize its performance, such as "time_gap", "loc_gap", "type_difference" between two aids.

## Ranking ##
- Features
1. session based: count, last action time_gap(click/carts/orders), last action type 
2. aid based: count, ratio(ratio of click, ratio of carts to orders...), time
3. session+aid based: count, last action time_gap(click/carts/orders), last action type
4. w2v embeddings: difference and similarity of candiates between sessions' history aids, then use mean\max\min\last\std
5. collaborative filtering score: collaborative filtering score of i2i, calculate the items between the candidates and user's historical aids.
6. count and ratio features of aids after/before diffrent windows(1day, 3day) of each sessions' last time

- Data Augment
1. use more weeks to train, we used 3 weeks for finnal submission.
2. use different seeds to generate more traindata(different cutoff of sessions), we used 5 seeds at last. 
those bring 0.001 boost on local cv.

- Model
1. lightgbm binary classifier, learning rate: 0.05, 1000 rounds
2. catboost binary classifier, learning rate: 0.05, 6000 rounds
enemble these two results scored LB:0.600.

## Interesting Findings ##
When optimize the task of orders, 
1. orders_prob = 0.7* orders_prob+0.2* carts_prob+0.1* click_prob, brings 0.0004 boost on orders' local cv(evaluation on the same session of three task).
2. My teammate rib concats carts and orders' training data, and add one feature(0, 1) whether the sample comes from carts or not, can get 0.0003 boost on orders' score.

## Enemble ##
For final submission, we used voting to enemble with Rib and ria's result(different candidats, LB: 0.601), LB goes to 0.602.
