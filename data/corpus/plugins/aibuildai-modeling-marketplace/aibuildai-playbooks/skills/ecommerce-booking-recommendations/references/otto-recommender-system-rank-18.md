# 20th place solution (Transformer inside)

Competition: otto-recommender-system
Rank: #18
Source: https://www.kaggle.com/c/otto-recommender-system/discussion/382924

First of all, I want to thank OTTO for organising great competition. It's was very challenging and clean :) 
Secondly, I want to thank all people who share their thoughts and ideas in kernels or forums. It helps a lot.
I will try to put my code on GitHub in a few days, need to clean up some deadline madness there.


## TL;DR

- use 3 types of co-occurence matrices (somewhat similar ideas to public notebooks, but different implementation and details)
- Bert MLM
- Matrix Factorization
- Catboost + PairLogitPairwise

### LB Progression (public)

- 0.576 - history + 60 candidates all-to-all co-occurence matrix + lightgbm ranker
- 0.579 - same, but 200 candidates
- 0.583 - added a bunch of item features (conversions, popularity etc.)
- **0.593 - optimized co-occurence matrix** (switched to sessions, added time weighing)
- 0.595 - add buy2buy features and Transfomer candidates, switch to catboost
- 0.597 - add buy2buy and type weighted co-occurence candidates, use more data for training (16/32 chunks)
- 0.598 - use full data for training
- 0.599 - use different candidates configs for clicks/buys, add MF candidates and scores
- 0.600 - use different transformers
- 0.601  - use x1.5 more candidates from each source for carts/orders


## Candidates retrieval

I use **max-recall@200** as a retrieval quality measure, but candidates from different sources can be more or less common with user history, so it seems more fair to outer join candidates with history to calculate max-recall. **0.598 LB is achievable only with co-visitation candidates**

I used different combinations of candidates for clicks and carts/orders.

```python
clicks:
    history_rank: 100
    cooc_rank: 200
    buy2buy_rank: 0
    cooc_tw_rank: 0
    mfc_rank: 100
    transformer_rank: 50
  carts/oders: # ~190 candidates on avg, 0.685+ WR
    history_rank: 100
    cooc_rank: 100
    cooc_tw_rank: 100
    buy2buy_rank: 100
    mfc_rank: 50
    transformer_rank: 50
```

### Co-visitation (all-to-all)

- Use actual user’s “sessions” - consecutive series of events, if there’s a gap > 900 seconds, it’s another session.
- Use exponential time weighing (more distant events are less significant)  0.99995^(abs(ts.x - ts.y))
- inverse rank weighing for user history events

Hyperparameters like session gap, time base and rank weight function were optimized with optuna, so **max-recall@200** was like **67.6** for this method.

### Co-visitation (type weighted)

1. Use 1 day gap with exponential time weighing (no “sessions”)
2. 10x weight for carts, 3x for orders

### Co-visitation (buy2buy)

1. Use 2 weeks gap, exponential time weighing
2. Use only carts and orders to calculate stats

### Transformer (small BERT)

It’s was hugely inspired by the [winning solution](https://github.com/Chubasik/yacup_recsys_2022) (by @chubasik) of recent Yandex.Cup Recsys track (I took [2nd place](https://github.com/greenwolf-nsk/yandex-cup-2022-recsys) there with classical 2-stage approach).

The idea is to train Masked Language Model, and then predict the “fake” last masked item in user session. Also, I fed action types (click, order, cart) as token_type_ids (which is mainly used for context separation in NLP tasks).

I trained MLM on train sessions, used 500k most carted items, **max-recall@200** was like **0.66.** This quality can be achieved with 3 epochs on full data, but it takes very long to train (7 hours on A100 GPU), so my experiments were very limited.

Adding this source of candidates gave boost of 0.002 in local CV, and this model score was second most valuable feature for ranking model. However, LB change was less then 0.001, and I spend last two weeks figuring out what went wrong. Finally, I trained two different models (train_no_val + val and train + test), this probably helped a little, but CV-LB gap was still bigger than before.
There's 3 epochs training from scratch, with metrics every 0.5 epochs
[3 epoch training (eval every 0.5 epochs)]



### Matrix Factorization

Matrix factorization on Pytorch with BPR-like loss and hard negative sampling. It achieves **0.665 max-recall@200**. There’s probably a huge space for improvement in weighing events by type and time and negative sampling strategies. Training time is **7h** in total for 20 epochs with AdamW optimizer. It almost made no difference to CV/LB score, but MF score was also one of the strongest features.


## Features

Best model uses around 200 features.

### User

- counters by type, normalized counter, time-based features
- number of “sessions”, avg session length
- avg/min/max/std “popularity” of item in user history

### Item

- item popularity by type - counters and ranks
- tried some derivatives to detect “trending” items, but they didn’t work for me
- item click/cart/order conversion rates

### User-Item

- interaction stats with item (number of clicks/carts, last timestamp)
- all features from co-visit matrices and statistics (mean/min/max/std for score/rank/normalized score)
- score from MF model
- statistics on MF item-item similarity with user history


## Ranker

I found out that `Catboost` with `PairLogitPairwise` loss is the best option for my data and final score is achievable without ensembling. Inference is fast (1-1.5h), but not as fast as LightGBM/XGBoost with cuml.ForestInference (thanks @buumoo for the clue).

Summary:
- 3-fold CV
- Catboost, PairLogitPairwise, 5000 iterations (2-4 minutes per fold on A100)
- separate models for each target (drop sessions w/o target + 20% random downsampling)
- LightGBM / XGBoost give slightly worse results (with lambdarank objectives), ensembling makes no differences

## Pipeline & technical details

I used mostly CUDF for data preparation and feature engineering. GPU memory is a bottleneck here, so I split data in 32 chunks.

From the beginning I tried to implement robust pipeline with DVC, and it worked well until the last days of the competition, when I decided to increase candidates count from avg 200 to avg 300 :)

One of the features of DVC is that it keeps track of parameters and changes, and stores results in cache. For example, if you want to experiment with some source of candidates, others won’t be recalculated. And you could return to previous state of data because of cache, but it’s not practical when you’re dealing with big amounts of data.
Here's how pipeline looks (below is almost the same part for test):



## Takeaways & fails

- try to keep data size as low as possible when actively trying ideas (firstly, I went from 60 to 200 candidates for 0.576 → 0.579 boost, then i went from 1/3 to full data for 0.596 → 0.597).
- as it’s multi-objective recommender system, I tried to use predictions of carts models as a feature for orders, but it did not work
- ensembling did not work after 0.6 LB. I used inverse rank averaging different rankers (e.g. catboost & lightgbm) on different candidates setups.
- computational and personal time investment in Transformer models was not great in terms of leaderboard score, but knowledge I got is priceless. Also, it was the first time I tried Weight & Biases for DL experiments tracking, and it’s awesome.
- this comp is hard, many ideas just don’t work. I think it took x3 time and effort than H&M with almost the same LB position
