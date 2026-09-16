# 19th place solution

Competition: data-science-bowl-2019
Rank: #19
Source: https://www.kaggle.com/c/data-science-bowl-2019/discussion/127396

First, I would like to thank the BS KIDS team, Kaggle team and all kagglers for this amazing competition. 

It was hard to handle qwk metric with so few test samples without fall in the public LB climbing! 

Here https://www.kaggle.com/steubk/dsb2019-bs-26-lgb-auc-xgb you can find  (100% kaggle kernel) solution.

This is a summary of my solution:

- The Magic
For those about to "hey, where is the kaggle's magic ?", this is my _magic blend_:

```y  = (2*0.7*y_lgb + 2*0.3*y_xgb + 2*3*y_auc_solved  + 3*y_first_try_success)/5```

with y\_lgb as lgb regression (rmse) , y\_xgb as xgb regression (rmse), y\_auc\_solved as binary classification(auc) for solved assessments (0/123) and y\_first\_try\_success as binary classifcation (auc) for first try success assessments (012/3) 

- Data Augmentation
thanks to @akasyanama13 and his kernel https://www.kaggle.com/akasyanama13/another-way-of-validation I added about 11% training samples (2.018/17.690) in public and many more (75% ?) in private.  

- Validation Strategy
For each learning model trained a 5 GroupKFold on installation_id, with fixed number of iterations (no early stopping), averaged with 5 seeds and validated with median of 5.000 shuffled truncated samples: this guaranteed the stability of the rmse and qwk on cv and qwk on private. 

- Feature enginering 
I generated more than thousand features (1.047), starting from public kernels (@artgor https://www.kaggle.com/artgor/quick-and-dirty-regression , @braquino https://www.kaggle.com/braquino/890-features  and others)
and then added features for:
current-assessment,  last-game, last-activity, last-assessment, "correct" event_data and encoded timestamp hour in cyclical continuous features.
 
- Feature selection 
I selected 128 features with RFE.
I done first elimation steps removing correlate features (corrcoef &gt; 0.9999) and features with minimal gain in lgb regression model. 
For last step (from 148 to 128) I computed rmse and qwk for each single feature and removed features for which the gain for rmse AND qwk was negative regardless of the lgb feature importance.
Last step gives an improvement for qwk in cv and private (cv: 0,5616 --&gt; 0,5624, private: 0.545 --&gt; 0.553)

- From RMSE to QWK (aka threshold definition)
I built a simple Bayesian optimization and validated with 5.000 shuffled truncated samples from train.

- The Magic Revisited (Simple models and ensembling)
This was the score before blending: 

| model | cv |cv std|(public)|private| 
| --- | --- |---|---|---|
|xgb|0.5574|0.008|0.544|0.550|
|lgb|0.5638|0.008|0.538|0.553|

A simple blend  ```(0.7*y_lgb + 0.3*y_xgb)``` of xgb and lgb model has given no imporvements on cv and public  but some improvements in private (0.555).

In the last days I tried some binary classifications with auc:
binary classification for solved assessments (0/123) and binary classification for  first try success (012/3)

The ordering induced by auc and the small number of elements for qwk made me think that I could try to blend the auc models directly with the regression models: In fact the blend  gives an improvement in cv (0.5680) and an improvement of 0.002 over the simple blend in private. (qwk final score:0.557)  


Thank you for reading !
