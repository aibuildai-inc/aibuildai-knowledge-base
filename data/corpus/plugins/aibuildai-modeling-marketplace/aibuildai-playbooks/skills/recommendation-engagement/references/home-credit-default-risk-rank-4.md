# 4th place sharing and tips about having a good teamwork experience

Competition: home-credit-default-risk
Rank: #4
Source: https://www.kaggle.com/c/home-credit-default-risk/discussion/64487

Congrats to all the winners.  we know we will move up in LB, but 4th place is probably the worst to be, however, our submission is our best cv and best lb. nothing to regret. 

my teammate will share more details and i will share what i learn.  i believe here are the few tricks that help us.

1. trend features, the last 1,3,5,10 in installments and pos,bureau features
2. train diverse models on subset of features and on overall big feature set  we had trained over 200 models on different parts with different models to make sure we completely squeeze all the values from our feature. Our final best single model has CV 0.8022 LB: 0.806  PB: 0.801 (gold medal!) the stacking CV is 0.8055 LB: 0.808 PB: 0.804
3. one trick we did was when we were running Bayesian to optimize the parameters, we save the predictions and use that as part of the oof
4. when build stacking layers, instead of using the 100+ oof, we first run a feature selection and then run the lightgbm, it helps improves the cv by about 0.0005

5. we applied a "manual" magic correction on our final results. this is an insight from my team mate greatdatanalyst that if you correct your prediction for revolving loan that is over 0.4 by 0.8, it will boost your auc.  maybe revolving loan has lower probability in the testset

tips about teamwork: i think we did really well on teamwork and everyone contributes and no one burns out. 

1. we have one person kain focusing on just running the stacking models and keep track of our progress. we have to create 2 slack pages because we are running out of space sharing files!

2. rest of the people keep creating features and generating oof, creating features and generating oof. towards the last week, everyone regroups and combine all the features and then run the models on the big dataset to squeeze the last bit of value


i have attached our best submission and feel free to ensemble and see what we can get :) 
keep kaggling.
