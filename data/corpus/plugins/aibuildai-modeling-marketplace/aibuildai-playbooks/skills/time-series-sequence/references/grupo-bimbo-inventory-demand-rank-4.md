# #6 Place Solution - Team Gilberto & Regis

Competition: grupo-bimbo-inventory-demand
Rank: #4
Source: https://www.kaggle.com/c/grupo-bimbo-inventory-demand/discussion/23232

First, congratulations to top 3 teams! It was a very hard problem and that placement is very well deserved.
That was a cool competition, thanks Kaggle and Bimbo for that experience!

Our solution uses week 9 for validation.
We trained more than 10 models, validated in week 9. But the final submission is basically a weighted average of 7 models.

Since Private LB is only in week 11 we don't have lag1 feature values directly available to use in the models (Probably that is the cause of many positions shakes in the end). So we built some models without lag1 features and some models using the predictions of week10 (Public LB) as the lag1 features of week11(Private LB). 
Models we trained:

- XGB( around +100 features: many combinations of lags)
- LR
- libFM(Sparse Features only)

Our best single XGB scored around 0.433x on Public LB  ;-D

Giba
