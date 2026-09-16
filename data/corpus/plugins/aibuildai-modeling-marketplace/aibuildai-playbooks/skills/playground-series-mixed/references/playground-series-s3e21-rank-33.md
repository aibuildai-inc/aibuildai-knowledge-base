# #33 Solution: Model distillation

Competition: playground-series-s3e21
Rank: #33
Source: https://www.kaggle.com/c/playground-series-s3e21/discussion/438745

My solution was pretty simple - is to select a few features, remove a few outliers and distill XGB model into RandomForest (given in the competition), which is discussed [here](https://www.kaggle.com/competitions/playground-series-s3e21/discussion/436434)

* In **feature selection**, I choose only `O2_1, O2_2, BOD5_5` columns to train both xgb and rf models, and added `o2_mean` feature only for xgb model, which is the overall mean of all o2_cols (including dropped cols like O2_3, O2_4 and etc.)

* In **removing outliers** I removed all datapoints containing `BSOD5_5` bigger than 40 and `NH4_5` bigger than 60 (even though `NH4_5` is dropped) for all xgb and rf training data, but only for xgb, i marked `O2_1, O2_2` cols which are less than 4 as *nan*.

* In **model distillation**, I first trained xgb model on its training data, and then used its **training** predictions as the training target for rf, and so that, rf must reatreat as much of xgb model efficiency as possible (In cross-validation, rf's been reatreating 88-90% efficiency). But also, we can think of distillation as a target normalizing, or, in other words, handling target outliers

And that's all, I also tried to use IsolationForest to remove outliers, but it gave me poor results

Good luck in the next competitions!
