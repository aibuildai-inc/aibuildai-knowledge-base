# 22nd place solution with code

Competition: equity-post-HCT-survival-predictions
Rank: #22
Source: https://www.kaggle.com/c/equity-post-HCT-survival-predictions/discussion/566769

Thank you Kaggle staff and competition organizers for such a interesting competition. Also, thanks to everyone for sharing during the competition. I learned lot of things in this competition from exploring different tabular model to ensembling them and about survival analysis data.

I have open sourced almost all my training notebooks in the GitHub account. here is the link : https://github.com/jaytonde/Kaggle-CIBMTR-2024

I have also made my final inference notebook public here : https://www.kaggle.com/code/jaytonde/hill-climbing-submission

I did not maintained the code well cause of shortage of time so reach out to me in case of any doubts. 

My solution is very simple yet generalized cause I do not have much gap in CV, LB and Private LB.
I have performed multiple experiments with different model and target preparation techniques and my final solution is just hill climbing of this experiments.


## Folding strategy
As per the discussions in the competition about folding strategies, I decided to do 10 random folds, as 5 folds stratified by race wasn't beneficial.


## Ensemble
Final solution have diverse ensemble of neural networks, XGBoost, CatBoost, LGBM, Lasso regression,  TabFPN, Support Vector Regression, Tablenet, torch surv,  Ridge Regression, TabM, Linear Regression, Table Transformer.

Following is my experiments with respective weights in hill climbing ensemble.
First best model having experiment name prlnn-exp-01 which is Refactoring Pairwise Ranking Network with classification mask and 0.2 shift.

| Experiment |  Description  | Weight |
|------------|--------------------|--------|
| catboost-exp-05 | CatBoost with Kaplan Meier targets and classification mask | 0.48000000000000087 |
| xgboost-exp-09 | XGBoost with Monotone constraints and  Kaplan Meier targets| 0.2700000000000007 |
| lgbm-exp-08 |  LGBM with BreslowFlemingHarringtonFitter targets | -0.20999999999999974 |
| catboost-exp-01 | CatBoost with Kaplan Meier targets | 0.10000000000000053 |
| lasso-exp-01 | Lasso regressor with Kaplan Meier targets | -0.03999999999999959 |
| nn-exp-04 | Neural Network having classification mask and 0.1 shift | 0.12000000000000055 |
| svr-exp-06 | Support vector regression with Nelson Aalen Fitter targets and classification mask| -0.0499999999999996 |
| ds-exp-01 | CoxPH model |  0.04000000000000048 |
| rf-exp-05 | Random Forest with Kaplan Meier targets and classification mask | -0.05999999999999961 |
| svr-exp-01 | Support vector regression with Kaplan Meier targets | -0.03999999999999959 |
| tf-exp-01 | tabpfn with Kaplan Meier targets | 0.04000000000000048 |
| catboost-exp-03 | CatBoost with Nelson Alen targets | -0.03999999999999959 |
| catboost-exp-06 | CatBoost with Nelson Aalen Fitter targets and classification mask | 0.0600000000000005 |
| tn-exp-02 | Tablenet | 0.03000000000000047 |
| xgboost-exp-02 | XGBoost with Cox Loss| -0.0499999999999996 |
| catboost-exp-04 | CatBoost with CoxPHFitter targets | 0.020000000000000462 |
| lgbm-exp-03 | LGBM with Nelson Alen targets| -0.029999999999999583 |
| ts-exp-01 | Torch Surv model | -0.019999999999999574 |
| xgboost-exp-10 | | 0.04000000000000048 |
| xgboost-exp-06 | XGBoost with Nelson Aalen Fitter targets and classification mask | -0.029999999999999583 |
| en-exp-02 | ElastiNet with classification mask | 0.020000000000000462 |
| ri-exp-06 | Ridge regression with Nelson Alen targets and classification mask | -0.029999999999999583 |
| tabm-exp-02 | TabM Model | 0.0600000000000005 |
| nn-exp-01 | Neural Network with 2 targets into 1 conversion | -0.05999999999999961 |
| lgbm-exp-01 | LGBM with Kaplan Meier targets | -0.0499999999999996 |
| nn-exp-06 | Refactoring Pairwise Ranking Network, thanks @albansteff | -0.0499999999999996 |
| xgboost-exp-05 | XGBoost with Kaplan Meier targets tuned parameters | 0.04000000000000048 |
| lir-exp-01 | Linear regression with Kaplan Meier targets  | -0.019999999999999574 |
| lgbm-exp-04 | LGBM with CoxPHFitter targets | 0.020000000000000462 |
| nn-exp-05 | Neural Network with some pre processing applied | 0.010000000000000453 |
| lgbm-exp-06 | LGBM with Nelson Alen Targets | -0.009999999999999565 |
| nn-exp-02 | Neural network with varied data pre processing | 0.010000000000000453 |
| xgboost-exp-01 | XGBoost with Kaplan Meier targets | 4.440892098500626e-16 |
| tt-exp-01 | Table Transformers Kaplan Meier Targets | 4.440892098500626e-16 |

For more details about model training and target preparation look out the corresponding notebooks in my GitHub repo. Experiment name is the notebook name in the repo.

## Final Submission CV and LB : 
| Description | CV | Public LB | Private LB |
| --- | ---------------------------------- |---------|--------|
|  Ensemble of above models with weights and rankdata | 0.688 | 0.693 | 0.694 |


## Acknowledgments
I would like to thank everyone who shared during the competition. I learned a lot and will try to apply all the learning in the next competition. Also, a special thanks to amazing notebooks by @cdeotte for his detailed discussions and starter notebooks.
