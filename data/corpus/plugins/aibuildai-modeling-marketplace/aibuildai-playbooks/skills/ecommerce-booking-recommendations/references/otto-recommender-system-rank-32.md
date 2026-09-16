# 34th (ex 37th) Place Solution (Polars is here to stay !)

Competition: otto-recommender-system
Rank: #32
Source: https://www.kaggle.com/c/otto-recommender-system/discussion/383843

Thank you to Otto and Kaggle for hosting this competition with such a challenging dataset.
A special thank to kagglers who shared their work making the competition even more stimulating  and - as usual - congratulations to the winners and everyone who enjoied the competition !

This competition introduced me to the Marlin daloader - a super efficient NVIDIA [dataloder ](https://github.com/NVIDIA-Merlin/dataloader) for recommender systems (thank you @radek1 !) and [Polars](https://www.pola.rs/):  a blazingly fast DataFrame library for huge datasets that natively supports multithreading and whose syntax is - for me - more intuitive than pandas. During the competition I was able to easily replace all the pandas code with polars code with a significant gain in performance (RAM and  CPU).

I think Polars it's a library that will become more and more important in the data ecosystem.

## My solution 

This is the visualization of the steps of my solution:



### Validation

1_000_000 random truncated sessions out of 1_801_251 sessions from 4th week

### Candidates Selection (Heuristic Model) (val:0.570, test:0.576)

Generated covisitation matrices for all aids, carts and orders and computed probability **P(aid,next-aid)** = **next-aid** follow **aid** in a (window of a) session containing **aid**  



Calculated the probability that an aid,cart,order is present more than once in a (window of a) session.

Heuristic model uses rules based on these two kind of probabilities.

### Base Ranker Model (0.573,0.580)

Selected first 100 candidates by heuristic model and generated 26 features:  

10 interaction (between candidate and session) based features of wich the most significative are: 
- *i_count*: occurences of candidate in session
- *i_self_aids_d*: sum of probabilities that the candidate is present more than once in the session 
- *i_sims_aids*: sum of probabilities that the candidate is a next-aid in the session
- ...

10 session based features : 
- *s_type_mean*: mean of clicks , cart and buys (see [link](https://www.kaggle.com/competitions/otto-recommender-system/discussion/379631#2108619))
- ...

6 candidated based features : 
- *a_self_buy_total*: occurences of orders for  candidate in all sessions  
- ...

### Feature engineering 1 (0.581, 0.588)

Added custom features for last 5 aids,carts,orders of each session (for a total of 81 feature):
- *i_aids_count_last_aid*: value of covisitation matrix  (last-aid-session,candidate)  
- *i_aids_total_last_aid*: occurence of last-aid in all sessions 
- *i_aids_count_last_aid/i_aids_total_last_aid*: probability that candidate follows last-aid in all sessions
- ... 

### Feature engineering 2: 0.583, 0.591

Generated covisitation matrices based on last two weeks (validation + test) and added corresponding features (137 features)

### Stacking : 0.585, 0.592 
Selected best 50 candidate from previous best model, added cross stacked predictions  from previous lgb models.
Added some interaction features based on word2vec model and used xgb as stacked model.
