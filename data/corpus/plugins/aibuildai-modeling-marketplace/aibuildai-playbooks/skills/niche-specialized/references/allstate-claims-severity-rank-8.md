# #8 solution

Competition: allstate-claims-severity
Rank: #8
Source: https://www.kaggle.com/c/allstate-claims-severity/discussion/26440

It seems the other top 10 had way more base models than I did. So I guess that was probably my mistake!

I had less than 20 base models (17 to be precise).  10 xgboost + 5 nn + 1 gbm + 1 lightgbm.  The best single model is xgboost with 1103 - 1104 public LB.

I used 5 fold instead of 10 fold to mainly save time training (probably another mistake!). My best single xgboost used eta = 0.0025, with about 13K trees. I would love to hear what others can get with higher eta. 

I found that 0.7 * FAIR + 0.15 * RMSE + 0.15 * MAE is a better loss compared to FAIR alone for xgboost (about 2 MAE better for same hyper parameters).

I mainly used the raw features + cat interactions. I also had two xgboost and one nn model trained on features where categorical are oof encoded based on loss.

And then one model each (xgboost, nn, gbm, lightgbm, random forest) for second layer.

Finally, quantile regression on first layer + second layer.


I didn't really have much "human time"  for this one, but I did have a new computer with some "computing time" so I let my computer run 24/7.

Thanks everybody. The forum discussion here is purely amazing!
