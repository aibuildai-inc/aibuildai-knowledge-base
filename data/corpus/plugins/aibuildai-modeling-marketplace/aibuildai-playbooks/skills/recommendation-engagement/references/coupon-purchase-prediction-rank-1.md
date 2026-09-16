# What's your guess about private LB?

Competition: coupon-purchase-prediction
Rank: #1
Source: https://www.kaggle.com/c/coupon-purchase-prediction/discussion/16736#93760

[quote=ssedhain;93754]

I guess the winning solution to be some variant of  Factorization machine !! My solution was completely based on learning a logistic regression per user .

[/quote]

Short version: 

I was doing probabilistic modelling of the following form: P(purchase) = P(user online) P(visit|online) P(purchase|visit).

- p(user online) was just a simple beta-bernoulli (this basically gives weights to coupons based on how many days they were displayed)
- other two were hierarchial logistic regressions. Inference by step wise doubly stochastic variational Bayes. (first learn the population parameters/distributions, hold them fixed and then find out user specific params. highly approximate, but seemed to work ok)
