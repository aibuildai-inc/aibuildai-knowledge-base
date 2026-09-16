# 9th place solution

Competition: porto-seguro-safe-driver-prediction
Rank: #9
Source: https://www.kaggle.com/c/porto-seguro-safe-driver-prediction/discussion/44700

Apologies for the delay -was flying ;)

This competition was another great example for why kaggle is...well great. There was so much sharing and valuable information in kernels/forums that I am sure, irrespective of our ranks, everybody learnt from it :)

Congrats to Jahrer for winning single-handedly the most popular kaggle competition ever!  congrats to all winners in general and utility for [a simple and neat 3rd place solution][1] . As he pointed out, you can find more information about [how to do well in competitions here.][2] 

Last but not least, I would like to thank my teamates Mario and Mathias for all their efforts in this competition .
Now that  credits are done (:)) , lets go to the solution.

cv strategy
===========

We spent half of this competition trying to figure out why the miss-match between train and test performance. In the end we may have lost valuable time with this, because all are models were inline with private leaderboard. In any case I can conclude that Strattified cv was somehow off than other approaches. We tried 2 other schemas:

 1. Random 5 fold-cv with a seed (=0, original order) that [Mario][3] pointed that results were consistent . After 5 folder were done, full model is trained on the whole data and test predictions are with that model. We would still see a varying up to -0.005 gap from LB for certain models that were consistent in private. 
 2. 20 kfolds with 2 splits each. We averaged the results of all these 20 fold runs. So test predictions were made by averaging (20x2) 40 models. These models had smaller gaps with NNs and bigger gaps with tree-based models at LB 

We saw  (the few) nns (we built) to score better in LB with (2), but tree-based models (like lightgbm and xgboost better with (1)

We built 2 ensembles, one with models of (1) and one with models of (2) and we (rank) averaged them

Model
===========
we recreated all popular (0280+) public kernels and fed them into these 2 cv schemas - there was no time for much improvisation. 

Sadly, we did not try many NNs until the last 2-3 days, so most of our scores are basically lightgbms and xgboosts.   

It seems that best transformations were dummies for categorical and everything else as is, target encoding was not adding much to this. For NNs, we managed to get one with 0.281 public LB after using these [Genetic algorithm features][4] (but removed all coefficients, divisions, tanh), using only the sums and a very simple 1-layer architecture with relu. Numerical features were scaled with Standard scaler and `np.log1p`

We briefly tried the NNs with embeddings , but did not work for us and did not have much time to tune them

in Addition to the public kernels, we were using other loss functions, specifically `regression`  with lightgbm. CV-wise, some of the regression functions (`huber, fair, regression`), were scoring better than standard `binary`

We built around 100 models with varying parameters, loss functions etc. Some with our own feature engineering, maybe 1 or 2 extra feature interactions , 

Some models came from StackNet too, for instance , the [libffm model][5] added well to the blend.

 Stacking
===========

Initially Nonlinear stacking was failing and a linear blender (after rank transformations) was giving better results, BUT , Faron did find a simple nn architecture that worked better and gave +.001 on top of the linear blender (consistent with LB, public and private). We built 2 ensembles, with same models (just different cv strategy ) and averaged the results. The gain was not too big in private LB, we should have explored different models instead of spending so much time there.

 


  [1]: https://www.kaggle.com/c/porto-seguro-safe-driver-prediction/discussion/44608
  [2]: https://www.coursera.org/learn/competitive-data-science/home/welc
  [3]: https://www.kaggle.com/mariofilho
  [4]: https://www.kaggle.com/scirpus/big-gp/
  [5]: https://github.com/kaz-Anova/StackNet/blob/master/parameters/PARAMETERS.MD#libffmclassifier
