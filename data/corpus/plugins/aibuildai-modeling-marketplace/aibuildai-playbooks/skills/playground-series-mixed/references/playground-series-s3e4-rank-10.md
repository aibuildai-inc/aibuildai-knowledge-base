# 10th place solution

Competition: playground-series-s3e4
Rank: #10
Source: https://www.kaggle.com/c/playground-series-s3e4/discussion/382539

HI folks, a bit surprised to jump up, here is what I did:
**Featrues:** 
- Used both competition and original datasets
- Transformed `Time` column to `Hour `and `Day`
- Taken division features from [this ](https://www.kaggle.com/competitions/playground-series-s3e4/discussion/381087) beautifull notebook.
- Dropped `Id` and `Time`.

**Model**
Catboost with custom Focal loss (link: [https://github.com/rahowa/catboost_focal_loss](https://github.com/rahowa/catboost_focal_loss))

**Cross validation**
Based on [this](https://www.kaggle.com/code/soupmonster/simple-lgbm-baseline-optuna) well-written notebook :
- 10-fold StratifiedKFold
- Predictions calculated by the model trained on each split and then avereged.

**Optimisation**
Optuna with TPESampler, it worked painfully slow, so only about 50 iterations. 
Hyperparameters optimised: `depth`, `learning_rate`, `l2_leaf_reg`, `subsample`, `min_data_in_leaf` and also  gamma parameter for focal loss.

*Please* upvote people whos work was used! 
This result might be pure luck, or these series inherently does not require overcomplication.
Thanks!
