# [1st place] Solution description

Competition: tabular-playground-series-may-2021
Rank: #1
Source: https://www.kaggle.com/c/tabular-playground-series-may-2021/discussion/243054

Hey folks, 

This is my first post on kaggle after 4 years. I never really participated since signing up until this year with the introduction of the Tabular playground series. These competitions prompted me to participate more and I have been having a lot of fun with these and learning a lot from the amazing people in this community sharing their work. I just personally need to get better at sharing and voting, so here is a start.

I am a bit surprised by this result as my solution is really boring and nothing fancy. I also am not convinced that this solution is a good classifier as it still is struggling with rare classes (ensembling across random seeds reduced that effect though.). I think what worked at the end of was trusting my local cv results and not using public lb and not blindly using the automl results. Don't get me wrong those are very helpful but in this competition they did not help my local cv so I did not use them at the end.

My workflow is in Google Colab and my code is quite messy and at time disorganized. With a very hectic day job, I did not have much time to clean things up but still thought to share in case someone finds any of this useful. Below is also a description of what worked at the end. 

[Here](https://colab.research.google.com/gist/academicsuspect/0aac7bd6e506f5f70295bfc9a3dc2250/tabular-may-baseline.ipynb?authuser=1#scrollTo=LtC_S97E8ep_) is the bulk of my code on Google Colab for individual models and cv (does not include the stacking stage). 

**Feature Selection/Engineering**
Nothing worked for me here. I tried PCA, clustering, also played with featuretools but all led to either horrible overfitting or poor local cv so at the end I did not pursue feature engineering much. 

For feature selection, due to the non existence of any dominant features or highly associated features with the target, I suspected that removing features would not be helping the cv so I kept all the features. 

For the categorical features, for my linear model (Logistic Regression) I used onehot encoding using scipy sparse features to fit in-memory, and for all the other models (all tree based) I just used OrdinalEncoder (essentially label encoding). 

**Base Models**
I ended up training 5 different models, each trained on a 5fold stratified cv on three random seeds, using out of fold predictions for each fold and averaging across folds for predicting the test set. The models were LogisticsRegression, RandomForest, XGBoost, LightGBM, and CATBoost. Weighting classes did not really work for LogisticsRegression and I used the defaults class weights in the rest of them.  I used Optuna to lightly tune each model (you will see my final parameters in the code above). 

**Over/Under Sampling**
Did not really try after reading the notebook from @remekkinas that pointed out it likely would not work. His analysis made sense to me and I did not have time, so did not really pursue any further.

**Ensembling**
Stacking worked quite well here. I tried weighted average of models and it did worse than stacking so I did not use that. I used a 5fold stacking of all models above and used a meta model of RidgeRegerssion, using CalibratedClassifierCV in sklearn. 

**Final submission**
I ended up clipping the probabilities (below 0.05 and above .95) to help the log_loss metric. [This  article](https://medium.com/@egor_vorobiev/how-to-improve-log-loss-score-kaggle-trick-3f95577839f1) explains very briefly (clipping helps mitigating the extremes of too small or too large values in the log_loss metric). Final submission had a public LB of 1.08564 and private of 1.08763. Looks like blending with top public notebooks could have brought the private LB scoe down to 1.08742 which I did not do.

**Credits**
Too many that I have lost count of. A couple of that have most directly used or learned from are as follows:
Faith @fatihozturk his 3rd place solution from Jan competition, gave me a robust CV framework which I have been using since and I also based my stacking off of his notebooks.
https://www.kaggle.com/c/tabular-playground-series-jan-2021/discussion/216087

Bizen @hiro5299834 every single notebook that he has shared in these series has been enormously valuable and insightful for me. Thank you for your generosity. 

@remekkinas as I used his analysis for over/under sampling
https://www.kaggle.com/remekkinas/tps5-is-about-sparsity-shap-extensive

@ryanbarretto    His notebook below has a solid stacking framework coded that I highly recommend. 
https://www.kaggle.com/ryanbarretto/boring-blend-of-stack-and-weights/comments
