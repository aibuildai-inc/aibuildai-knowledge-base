# A 10th Place Experiment

Competition: playground-series-s5e11
Rank: #10
Source: https://www.kaggle.com/c/playground-series-s5e11/writeups/a-10th-place-experiment

This contest was a test of two ideas by @Tilii and @CDeotte. Create models that are uncorrelated and do not be afraid of removing columns in your models. Talking about feature engineering,there are so many different ways too try. I mainly used @CDeotte ideas.In essence, I did many of the same feature generations as those above and below me in the contest. I created about 24 models with various features and tried to assemble them just using roc score and not LB score. I had 24 models with only 3 correlated with some of the other models. The best ensemble was with the hillclimbers  program including the 3 correlated models. The results are shown below: 
Models to be ensembled | (24 total): 

xgb_pseudo_recip:              0.92781 (best solo model)
xgb_pseudo:                    0.92780
lgb_pseudo_recip:              0.92777
AG3:                           0.92772
xgb_new:                       0.92770
xgb_base_ln_0.9276594:         0.92766
lgb_0.927645:                  0.92765
lgb_del1:                      0.92762
AG2:                           0.92761
RealTD1:                       0.92760
cat_2:                         0.92758
cat_base:                      0.92750
TD_Essembly2:                  0.92749
RealTD2:                       0.92746
TD_Essembly:                   0.92739
xgboost:                       0.92734
cat_TD:                        0.92732
ydf:                           0.92716
xgb_base_ln_0.9269135:         0.92691
xgb_0.9269169:                 0.92690
ag:                            0.92456
lgb_del2:                      0.87601
xgb_drop_debt_to_income_ratio: 0.87436
xgb_drop_employment_status:    0.82128

[Data preparation completed successfully] - [Initiate hill climbing] 

Iteration: 1 | Model added: AG3 | Best weight: 0.50 | Best roc_auc_score: 0.92810
Iteration: 2 | Model added: ag | Best weight: 0.07 | Best roc_auc_score: 0.92813
Iteration: 3 | Model added: RealTD1 | Best weight: 0.16 | Best roc_auc_score: 0.92816
Iteration: 4 | Model added: xgb_base_ln_0.9269135 | Best weight: 0.08 | Best roc_auc_score: 0.92817
Iteration: 5 | Model added: lgb_del2 | Best weight: 0.01 | Best roc_auc_score: 0.92817
Iteration: 6 | Model added: xgb_drop_employment_status | Best weight: 0.01 | Best roc_auc_score: 0.92818
Iteration: 7 | Model added: xgb_new | Best weight: -0.12 | Best roc_auc_score: 0.92819
Iteration: 8 | Model added: lgb_del1 | Best weight: 0.09 | Best roc_auc_score: 0.92819
Iteration: 9 | Model added: xgboost | Best weight: -0.07 | Best roc_auc_score: 0.92820
Iteration: 10 | Model added: xgb_pseudo | Best weight: 0.06 | Best roc_auc_score: 0.92820
Iteration: 11 | Model added: TD_Essembly2 | Best weight: -0.07 | Best roc_auc_score: 0.92820
Iteration: 12 | Model added: cat_2 | Best weight: 0.02 | Best roc_auc_score: 0.92820
Iteration: 13 | Model added: cat_TD | Best weight: -0.04 | Best roc_auc_score: 0.92820
Iteration: 14 | Model added: xgb_0.9269169 | Best weight: -0.02 | Best roc_auc_score: 0.92821
Iteration: 15 | Model added: cat_base | Best weight: 0.02 | Best roc_auc_score: 0.92821
Iteration: 16 | Model added: xgb_base_ln_0.9276594 | Best weight: -0.03 | Best roc_auc_score: 0.92821
Iteration: 17 | Model added: lgb_pseudo_recip | Best weight: 0.03 | Best roc_auc_score: 0.92821
Iteration: 18 | Model added: lgb_0.927645 | Best weight: -0.03 | Best roc_auc_score: 0.92821
Iteration: 19 | Model added: RealTD2 | Best weight: 0.02 | Best roc_auc_score: 0.92821
Iteration: 20 | Model added: TD_Essembly | Best weight: -0.02 | Best roc_auc_score: 0.92821
Iteration: 21 | Model added: AG2 | Best weight: -0.01 | Best roc_auc_score: 0.92821
Which gave a 10th place finish.
