# A quick summary of Team Mystic (3rd place) solution

Competition: grupo-bimbo-inventory-demand
Rank: #3
Source: https://www.kaggle.com/c/grupo-bimbo-inventory-demand/discussion/23633

We use data from week 3 - 6 to generate features for week 7, and week 3 - 7 to generate features for week 8 - 9, and then train on week 7 to predict on week 8, and 9. Similarly, we use data from week 3 - 8 to generate features for week 9, and then train xgboost on week 9 to predict week 10 & 11.

We have two parts of features:

1.  Shallow features. We calculate max, min, median, mean, and standard deviation of demand on interactions of features. We also include features (Venta_uni_hoy, Venta_hoy, Dev_uni_proxima, Dev_proxima, Demanda_uni_equil) of the last 3 weeks.

2.  Deep features.  We train a dozen of linear models to get "deep" interaction signal between features. Vowpal wabbit, FTRL implemented with Pypy, libFM, and libFFM are all used to generate deep features.

Combine these two parts, one single xgboost model is good enough to get the score we have in the public LB.

Please comment if you have any questions. And code with detailed doc will be linked here after we get the final approval from kaggle and grupo bimbo. :)
