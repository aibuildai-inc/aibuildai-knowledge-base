# 6th place solution (single model LB 0.603)

Competition: otto-recommender-system
Rank: #6
Source: https://www.kaggle.com/c/otto-recommender-system/discussion/384120

Thanks to kaggle and OTTO for the great game. This is my first solo gold medal and I'm very excited about it.
This is my overall model framework.


# Retrieval
I've included three recalls
- top 150 Co-visitation Matrix by CHRIS DEOTTE [https://www.kaggle.com/code/cdeotte/candidate-rerank-model-lb-0-575](url)
- top 100 click 2 click bidirection i2i similarity with pos, time, session, aid weight 
- top 100 click 2 cart bidirection i2i similarity with pos, time, session, aid weight 


# Features
- session feats: 
1. the counts and frequency of user clicks/orders/carts
1. user last clicks/orders/carts aid and hour
1. user last behavior type
- aid feats:
1. aid clicks/orders/carts counts
1. aid clicks/orders/carts ratio
1. aid clicks/orders/carts time
1. aid behavior mean type
- session aid feats:
1. user clicks/orders/carts aid counts
1. user clicks/orders/carts aid time
1. user behavior aid mean type and last behavior type
1. abs(hots/time of user click/cart/order the aid  - aid click/cart/order hots/time)
- sim feats:
1. Co-visitation Matrix rank
1. clicks/carts/orders to clicks/carts/orders 2 clicks/carts/orders i2i/i2i2i sim with pos, time weight.
1. clicks/carts/orders to clicks/carts/orders 2 clicks/carts/orders i2i/i2i2i mean/max/min/std/last sim 
1. clicks/carts/orders to clicks/carts/orders aid pair sim with pos, time weight.
1. clicks/carts/orders to clicks/carts/orders aid pair mean/max/min/std/last sim 
1. w2v embedding sim 

# Train And Validation 

In the verification phase, I used Radek’s CV strategy.
For online prediction, train_v1 + train_v2 + valid was used as training data.

# Model
I used lightgbm binary classifier, learning rate: 0.02, 5500 rounds

# Local CV And LB Score
best single model local cv parts:
1.orders recall@20 is 0.6715
1.carts recall@20 is 0.4433
1.clicks recall@20 is 0.5561
My local cv is 0.6715 * 0.6 + 0.4433 * 0.3 + 0.5561 * 0.1 = 0.5915
LB is 0.60335

# Enemble
I didn't run the second model，i am using the previously submitted version of the model for a blend of probability. this is give me the final score 0.60341
