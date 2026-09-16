# 1st Place Solution for the Regression with an Abalone Dataset Competition

Competition: playground-series-s4e4
Rank: #1
Source: https://www.kaggle.com/c/playground-series-s4e4/discussion/499174

Thanks for hosting the Playground series, it's a great place for experimenting and focusing on model building 👏🏽

# Context
Business context: https://www.kaggle.com/competitions/playground-series-s4e4
Data context: https://www.kaggle.com/competitions/playground-series-s4e4/data

# Overview of the Approach
With this competition, I tried to max out the heavy ensembling GBDT strategy. In an earlier tabular data competition I tried different neural network architectures only to observe that none of them is competitive. I guess the papers claiming ANNs are finally on a par with GBDTs for tabular data haven't done proper preprocessing and fine-tuning. So here I focused mainly on GBDTs. AutoGluon indirectly adds some ANN to the ensemble, though.

The solution in a nutshell:
- Feature Engineering using an AutoML solution (OpenFE, [Zhang et al. (2022) OpenFE: Automated Feature Generation with Expert-level Performance](https://arxiv.org/abs/2211.12507), https://github.com/IIIS-Li-Group/OpenFE)
- Selecting ca. 20 features (in addition to the original columns) using Sequential Feature Selection, individually for LGBM, XGB, Catboost. Examples: 'freq(Shell_weight)', '(Length-Shell_weight)', '(Whole_weight/Shucked_weight)', 'freq(Whole_weight)', '(Whole_weight+Shell_weight)', '(Length/Shell_weight)', 'residual(Whole_weight)', 'log(Whole_weight)', 'max(Whole_weight,Height)'
- Find some good hyperparam combinations using Bayesian Optimization (WandB Sweeps) for LGBM, XGB, CatBoost, HistGB and RandomForest Regression. Attempts with custom MLP were quickly abandoned, see my comment above 😞
- Train those models (10-folds CV), save OOF predictions. Use complete original Abalone dataset for training in each fold, never for validation.
- AutoGluon (first I had some failed attempts that exceeded the Kaggle time limit or didn't return proper OOF predictions)
- Two-step ensemble:
1. The ensemble itself consists of 49 models. I optimized the weights using the OOF predictions and Nelder-Mead algorithm. The best results included negative coefficients and summed up to 0.997. That seemed cheesy to me but consistently outperformed coefficients that were clipped at min=0.0 and scaled to sum up to 1.0. Thus, I kept the optimized coefficients to blend the models' predictions.
2. Tried adding some well-performing public notebook submissions to the ensemble. In the absence of OOFs I used some submissions against public LB to find an optimal weight of 17% and ended up with using only one ANN-based submission by [endofnight17j03](https://www.kaggle.com/code/endofnight17j03/abalone-age-ann-xgboost-catboost) 👋 

My second submission skipped the external public notebook and scored 0.14372/0.14379 (public/private LB), i.e. it would have finished first, too.

# Details of the submission
## Handling of the RMSLE metric 
...by using proper loss fn. and eval. metric if possible:
- LGBM: custom MSLE loss (similar to proposal by [broccoli beef](https://www.kaggle.com/siukeitin) in this [discussion](https://www.kaggle.com/competitions/playground-series-s4e4/discussion/488283) 👋) - btw 'gamma' objective scored only slightly worse
- XGB: 'reg:squaredlogerror'
- np.log1p / np.expm1 for other models

## Best individual models with CV RMSLE:
- LGBM: 0.14611
- CatBoost: 0.14620	
- XGB: 0.14616 
- XGB Classifier with Regression Head: 0.14680
- HistGB: 0.14648
- (AutoGluon: 0.14592)
- (Ensemble: 0.14514)

## Classification with custom Regression Head
[Prajwal Anagani](https://www.kaggle.com/inagana) introduced an unusual [approach](https://www.kaggle.com/code/inagana/ps4e4-classification-with-xgboost-0-148)👋: Using an XGB Classifier with 'multi:softprob' objective but adding a Softmax-based regression head. I still don't know why it works (maybe due to the synthetical nature of the dataset and the discrete regression target) but the optimized ensemble weights include it despite worse individual CV results in comparison to (traditional) XGB regressors.

## What did not work
Polynomials, more sophisticated stacking/blending/meta-model approaches, classic ML approaches, Stratified KFold, any kind of encoding of 'Sex' feature (except OHE for models that don't accept categorical features natively).

## High Value Targets
My models never predicted more than a maximum of 20 Rings (with 29 being the maximum in the train dataset). Any approach to tackle that gap (SMOTE, augmentation) worsened CV significantly. So did excluding the outlier samples from training.

# What I learned in the competition: 
- AutoML approaches work (AutoFE, AutoGluon) but require manual post-processing to achieve competitive results. 
- The Kaggle API is awesome for writing code locally in PyCharm and having it executed at Kaggle with a few clicks.
- Wrap often-used code (e.g. Trainer for CV and prediction) into a custom Python Package to guarantee comparable results when experimenting. Kaggle API helps here, too, to automate deployment.

Thanks to the authors and other Kagglers who shared lots of great insights👍
