# 13th Place Gold Solution

Competition: amex-default-prediction
Rank: #13
Source: https://www.kaggle.com/c/amex-default-prediction/discussion/348014

First of all, we would like to thank kaggle and the staff for hosting such a great competition. We really miss interesting tabular challenges these days.

# 1. Summary 
 Our solution is based on extensive data cleaning and multi-model weighted average and stacked ensembles using ranked probs. Using extensive data cleaning, our single model was boosted to cv 0.0004~0.0008 compared to the public Raddar's dataset.The final solution (PVT 0.80842/PUB 0.80105) which shaked up to gold zone  is the average of 3 ensemble models:
- LGBM stack with 61 models
- CMA(Covariance Matrix Adaptation) weighted average with 54 models
- Weighted average with 55 models

# 2. Extensive data cleaning 
Starting from Raddar’s (thanks @raddar) preprocessed dataset, some other features were cleaned, added and modified in order to build a second version of the dataset. The idea here was to clean the data even more and add some diversity to the ensemble. Some clusters of clients were added based on the pattern of missing variables for each set of features: B, D, P, R and S. Also, some continuous variables showed a curious noise pattern. Looks like a uniform noise was added in the range of (0 - 0.01] of some variables. Take a look at feature B_11 distribution. White noise was clearly added just in the 0.01 range.





Spent some time trying to figure out a way to remove that noise and found some combinations of filters using other features that worked pretty well. For example, the B_11 feature can be cleaned by using a filter in B_1. Taking indices when B_1 is in range of 0-0.01 and inverting the signal of B_11 based on these indices, the new histogram becomes:



This kind of cleaning was made with some other features like: B_1, B_5, B_7, B_11, B_15, B_17, B_18, B_21, B_23, B_24, B_26, B_27, B_29, B_36, B_37, D_58, D_60, D_69, D_71, D_102, D_133, D_144, R_1, R_6, S_16, S_17, S_19, S_22 and S_27. Extra feature cleaning helped to boost GBDT scores and added some diversity when stacking with other models.





## 2.1 Example of boosting cv using cleaning data
- LGBM : 0.7976 → 0.7983 ( +0.0007 )
- XGB : 0.7978 → 0.7986 ( + 0.0008 )
- CatBoost :    0.7964 → 0.7968 ( + 0.0004 ) 

 ※ clean data plus minor change 
   - LGBM 5kfold → 15kfold (+0.0013)
   - CatBoost longer earlystop (+0.002 )


# 3. Single model 

## 3.1 Features and modeling

Basically, we used the features and the models of public notebooks.
Thank you for @thedevastator, @ragnar123, @ambrosm, @cdeotte, @roberthatch

Note: some of the 1st level models don’t include features (B_29, R_1, D_59) from adversarial validation.

## 3.2 Representative best single model in each model type 

| model       | features | kfold | cv     | public lb | private lb | comment                      |
|-------------|----------|-------|--------|-----------|------------|------------------------------|
| LGBM        | 1784     | 15    | 0.7991 | 0.79961   | 0.80679    | dart with early stop         |
| XGB         | 3413     | 5     | 0.7986 | 0.79873   | 0.80662    | pyramid [REF](https://www.kaggle.com/code/roberthatch/xgboost-pyramid-cv-0-7968)                      |
| CATBoost    | 2290     | 5     | 0.7968 | 0.79832   | 0.80629    |                              |
| MLP         | 446      | 5     | 0.7919 | 0.79102   | 0.79993    | KD using LGBM oof prediction |
| GRU         | 188      | 5     | 0.7948 | 0.79418   | 0.80363    | KD using LGBM oof prediction |
| Transformer | 188      | 5     | 0.7932 | 0.79498   | 0.80379    | KD using LGBM oof prediction |




# 4. Ensemble 

We used the two methods for ensemble with ranked probs. One is the LGBM stacking, the other is the CMA (Covariance Matrix Adaptation) Evolution Strategy
[REF](https://www.scm.com/doc/params/python/optimizers/cmaes.html)


And final submission is the average of the following 3 ensemble models(case1～3) ::
|Case-Method|CV|public|private|#GBDT|#1/2dcnn|#MLP|#TCN|#GRU|#Transf|
|------------|----------|-----------|-----------|-----------|----------|----------|--------|------------|
|1-LGBM stacking| 0.8010 | 0.80084 | 0.80882    | 48 |2            | 5          | 2          | 3          | 1|
|2- cma             | 0.8032 | 0.80086 | 0.80794    |  41 | 2            | 5          | 2          | 3          | 1|
|3- cma             | 0.8028 | 0.80068 | 0.8079     |  49 | 0            | 3          | 0          | 3          | 0|

As you can see, the public lb is almost the same, but the private lb shows much better LGBM stacking. We would have been hard pressed to get the gold medal without LGBM stacking.

## 4.1. Final sub (ensemble results of case 1 ~ 3)
The ensemble result of the above case 1-3 is our final submission.
public lb 0.80105, private lb 0.80842 (13th).

※ Note that : LGBM stacking alone (0.80882) has the capability of ranking 6th.

# 5. Relationship of cv and lb 

As for reference, we share the relationship of cv and lb. It is easy to see without looking at the plot with cma, but the lgbm stacking is a clean extension of the cv/lb straight line of the single model to some extend (Maybe cma is overfitting…)



# 6. Other tips that worked well 

- knowledge distillation
- longer early stop (3000～10000)
- large kfold
- full training
- pseudo labeling


# 7. Didn't work well 
- post process
- using dow average data
- changing weights of samples in LGBM
- adding focal loss to LGBM
- optimizing 0.4G+0.6D, 0.8G+0.2D, etc


# 8. Late submission result
 After the end of competition we realized that we had 2 ensembles with PVT LB > 0.80920 (3-4th place) - however, were not considered due to lower CV scores.

 1) avg rank of 3 stack models (XGB+LGB+Catboost) - PVT 0.80928, PUB 0.80067, CV 0.8008

 2) LGB stack with 61 models + meta features (top 50 eng. features) - PVT 0.80926, PUB 0.80072, CV 0.80097

# 9. Team organization 
We used github for code storage, wandb for experiments tracking and kaggle datasets for OOF storage and sharing with the team.

Team members: 
https://www.kaggle.com/titericz
https://www.kaggle.com/yamsam
https://www.kaggle.com/imeintanis
https://www.kaggle.com/chumajin
https://www.kaggle.com/kurokurob
