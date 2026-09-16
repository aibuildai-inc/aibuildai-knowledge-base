# 13th Place Solution

Competition: otto-recommender-system
Rank: #13
Source: https://www.kaggle.com/c/otto-recommender-system/discussion/383229

First of all, thank you for organizing a great competition. I would also like to thank everyone who participated in the competition with us and the four of us who worked together as a team.

Sepcially thanks to my teammates: @trasibulo @virilo and @albert2017 

## Candidates Selection

We merge several types of models for gettint the final candidates. Each model contribute with different number of candidates and then we ensemble them over a ranking based weights to get the final 150 candidates. We tested 100 and 150, we noticed a slightly improvement with 150.

### Scores: 

Our final top150 recall was: <br>
    clicks recall=0.7241 <br>
    carts recall=0.5636 <br>
    orders recall=0.7399 <br>
    **Overall Recall(top 150)** = 0.6854 <br>

### Base models

- historical items weighted by time and type.
- fnoa candidates (50-150 candidates depending on the historic).
- transformers4rec based model in order to get candidates. (100)
- short term coocurrence (100)
- covisitation matrix (100)
- w2vec knn (100)

## Ranking

### Fnoa 

I started with pandas but finally I changed all my pipeline to polars.

I decided to do one model per item type: clicks, carts, orders

#### Features used:
item_features
session_features
user_items_features
features from candidates generation (probs, similarity, etc..)

#### Models:
We tried lgb and catboost;
Although lgb was slightly better on LB, catboost was faster so at the end I decided to go only with catboost.

#### Things that didnt work:
I tried many things for improving validation score at the end but everything i tried didn't seem to help
Use carts predictions as feature for orders
Stacking
Different models
Train with all data
w2vec similarity features

### Xiao
Xgboost model and lgbm model. (comming soon)

### Virilo
BST Transformer.
https://www.kaggle.com/competitions/otto-recommender-system/discussion/384441

## Ensemble & Stacking
We tried both ensemble and stacking.

**Stacking:** we didnt managed to make it work on LB (while we were getting a good validation score, it was not reflected on LB)

**Ensemble:** it seemed to be more stable but also didnt give us the expected boost.

At the end our final submission was an ensemble of probabilities from 3 models: xgb,lgb,catboost

LB results-> single model (0.602 low); ensemble (0.602 high)
