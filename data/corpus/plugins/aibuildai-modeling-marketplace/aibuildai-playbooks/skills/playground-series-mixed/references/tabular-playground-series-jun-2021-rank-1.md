# 1st place solution: everyone can be a winner!

Competition: tabular-playground-series-jun-2021
Rank: #1
Source: https://www.kaggle.com/c/tabular-playground-series-jun-2021/discussion/250046

0
Preface words:
If one has practiced tps1-5 seriously , everyone can tackle this problem successfully. 



1
Step 1, used the following  3 sets of parameters to build about 15 xgb models, chose about 90 predictions as features to train nnet


1.1
params <-list( 
  "tree_method"         ="hist",
  "max_bin" =512,   
  "max_leaves"=150,
  "min_child_weight"    =110,
  "grow_policy"="lossguide",
  "eta"                 = 0.009,
  "max_depth"           =0,
  "subsample"           = 0.7,
  "colsample_bytree"    = 0.11
  ,"colsample_bylevel"   = 0.90
  #,"colsample_bynode"    = 0.80
  ,"lambda"              =0
  ,"alpha"=22
  ,objective = "multi:softprob" 
  ,eval_metric ="mlogloss"
  ,"num_class" =9
  ,"max_delta_step"=10)

1.2
params <-list( 
  "eta"                 = 0.006,
  "max_depth"           =22,
  "min_child_weight"    =110,
  "gamma"               =0.01,
  "subsample"           = 0.7 
  ,"colsample_bytree"    = 0.1
  ,"colsample_bylevel"   = 0.90
  ,"colsample_bynode"    = 0.80
  ,"lambda"              =1.5
  ,"alpha"=21
  ,objective = "multi:softprob" 
  ,eval_metric ="mlogloss"
  ,"num_class" =9
  ,"max_delta_step"=10)

1.3
params <-list( 
  "tree_method"         ="hist",
  "max_bin" =512,   
  "max_leaves"=200,  
  "grow_policy"="lossguide"
  #"max_depth"            =3
  ,"min_child_weight"    =110
  ,"eta"                 =1
  ,"alpha"               =22
  ,"lambda"              =0 
  ,"subsample"           = 0.70  
  ,"colsample_bytree"    = 0.11
  ,"colsample_bylevel"   = 0.90
  ,"num_parallel_tree"   =110
  ,objective = "multi:softprob" 
  ,eval_metric ="mlogloss"
  ,"num_class" =9
)

2
Step2:  ensemble

R  package of ann2
bst<-neuralnetwork(X,Y,hidden.layers = c(63,27),standardize=TRUE,
                   optim.type = 'adam', learn.rates = 0.0004, val.prop =0.2
                   ,batch.size=320,random.seed=8888688
                   ,L1=2,L2=0,activ.functions=c('sigmoid','sigmoid')
                   ,n.epochs =200)

To this step, if only using tree prediction as features,  private score may be 1.73900;
If adding some nnet prediction as features, private score may be 1.73890


3
Step3: weighted average



Thanks,Kaggle!
