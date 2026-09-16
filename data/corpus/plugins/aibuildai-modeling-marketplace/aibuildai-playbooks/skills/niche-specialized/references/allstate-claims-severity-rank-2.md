# #2nd Place Solution

Competition: allstate-claims-severity
Rank: #2
Source: https://www.kaggle.com/c/allstate-claims-severity/discussion/26427

Hi! 

Thanks for all participants and especially forum/script authors, it was a very fun competition and I learned A LOT!

I ended in 9th place in public LB, but moved to 2nd in private which was a pleasant surprise =)

Here is my solution description. As most of solutions, I suppose, it's based on building a lot of different models and stacking.

**First-level models**

I've trained a lot of diverse models on top of different data transformations, target transformations and different parameters.

The main ones were XGB and Keras NN (all of them with 4-6 bags) which provide me with following best single model results:

Best XGB - 1122.64977 CV, 1105.01969 Public LB, 1116.74215 Private LB

Best NN - 1130.29286 CV, 1111.90884 Public LB, 1122.06610 Private LB

Also, there was [LightGBM][1] (which is pretty fast, but seem to provide less tuning options then XGB, I've used [pyLightGBM][2] for Python integration), sklearn models (Random Forest, Extra Trees, Gradient Boosting, Linear regression, Support Vector Regression and KNN) and LibFM.

As SVR and KNN took a lot of time to train, I've trained them on small data samples and then averaged results.

In total there were about 70 L1 models which I used in my solution.

**Data transformations**

I've used different categorical encodings in different models - lexical, dummy, bayes, the best ones being lexical for tree-based models and dummy for other ones.

I've tried applying SVD to numeric + dummy categoricals and it seem to help for non-tree models (for example, CV of libfm model with SVD improved from 1196 to 1177).

Also, what helped for non-tree models is clustering data and then transforming it to cluster-distance space and applying RBF function. It was especially useful for linear regression (improved from 1237 to 1202), but also good for NN models.

In my late XGB models I also used categorical combinations, the best way to select combinations was using excellent [Xgbfi][3] tool - just take dump of good XGB model without combinations, run Xgbfi and take some number of 2-way interactions.

**Parameter tuning**

To select parameter sets I've used [BayesOptimization][4] package for smaller models (like RF and ET), but XGB and NN were tuned manually - there were several good links in the forums which provided me with some intuition on how to tune XGB params (especially [this][5] one). 

**Second level**

In second level I've mainly trained XGB and Keras NN models, with different params, but also included linear regression with different target transformations, random forests and [gradient boosting from sklearn][6], because it can optimize MAE directly.

In total I've trained around 50 L2 models, which were futher groupped and averaged (see below).

**Third level**

For my third level I used [quantile regression from statsmodels package][7]. It doesn't have any regularization and seems to produce noisy results, so in last couple of days I tried to reduce model noise:

* I've trained L3 model with 8 bags in each fold and used fold-average predictions for submission.
* I've groupped L2 models by their similariity and averaged predictions in each group, so producing 10 features for L3 level. Also I've applied power correction to some of groups, like X^1.03.


Also, in the end I've selected around of ten my best submissions from different time and averaged them, what gave me additional 0.1 in public LB.


Thats all.  I'll post a link to repository with all the code in a day or two (though it may be a bit messy) (UPD:  https://github.com/alno/kaggle-allstate-claims-severity).

Thanks again for competing!


  [1]: https://github.com/Microsoft/LightGBM/
  [2]: https://github.com/ArdalanM/pyLightGBM
  [3]: https://github.com/Far0n/xgbfi
  [4]: https://github.com/fmfn/BayesianOptimization/
  [5]: https://www.kaggle.com/c/santander-customer-satisfaction/forums/t/20662/overtuning-hyper-parameters-especially-re-xgboost?forumMessageId=118487
  [6]: http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html
  [7]: http://statsmodels.sourceforge.net/devel/examples/notebooks/generated/quantile_regression.html
