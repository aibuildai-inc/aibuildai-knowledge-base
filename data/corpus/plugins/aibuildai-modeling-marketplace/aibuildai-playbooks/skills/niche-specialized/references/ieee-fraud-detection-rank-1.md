# 1st Place Solution - Part 2

Competition: ieee-fraud-detection
Rank: #1
Source: https://www.kaggle.com/c/ieee-fraud-detection/discussion/111308

In "1st Place Solution - Part 1" posted [here][1], we discussed the benefits of classifying **clients** (credit cards) instead of **transactions** in Kaggle's Fraud competition. Here we will discuss the technical details. 

# Final Model
Our final model was a combination of 3 high scoring single models. CatBoost (Public/Private LB of 0.9639/0.9408), LGBM (0.9617/0.9384), and XGB (0.9602/0.9324). These models were diversified because Konstantin built the CAT and LGB while I built the XGB and NN. And we engineered features independently. (In the end we didn't use the NN which had LB 0.9432). XGB notebook posted [here][8].

One final submission was a stack where LGBM was trained on top of the predictions of CAT and XGB and the other final submission was an ensemble with equal weights. Both submissions were post processed by taking all predictions from a single client (credit card) and replacing them with that client's average prediction. This PP increased LB by 0.001.

# How to Find UIDs
We found UIDs in two different ways. (Specific details [here][9]).
* Wrote a script that finds UIDs [here][3]
* Train our models to find UIDs [here][4] and [here][8] 
  
If you remember, Konstantin's original public FE kernel [here][5] without UIDs achieves local validation AUC = 0.9245 and public LB 0.9485. His new FE kernel [here][4] achieves local validation AUC = 0.9377 and public LB 0.9617 by finding and using UIDs. Soon I will post my XGB kernel which finds UIDs with even less human assistance and proves to beat all other methods of finding UIDs. (XGB posted [here][8]). The purpose of producing UIDs by a script was for EDA, special validation tests, and post process. We did not add the script's UIDs to our models. Machine learning did better finding them on its own.

# EDA 
EDA was daunting in this competition. There were so many columns to analyze and their meanings were obscured. For the first 150 columns, we used Alijs's great EDA [here][6]. For the remaining 300 columns, we used my V and ID EDA [here][7]. We reduced the number of V columns with 3 tricks. First groups of V columns were found that shared similar NAN structure, next we used 1 of 3 methods:
* We applied PCA on each group individually
* We selected a maximum sized subset of uncorrelated columns from each group
* We replaced the entire group with all columns averaged.
  
Afterward, these reduced groups were further evaluated using feature selection techniques below. For example, the block `V322-V339` failed "time consistency" and was removed from our models.

# Feature Selection
Feature selection was important because we had many columns and preferred to keep our  models efficient. My XGB had 250 features and would train 6 folds in 10 minutes. Konstantin will need to say what his models had. We used every trick we knew to select our features:
* forward feature selection (using single or groups of features)
* recursive feature elimination (using single or groups of features)
* permutation importance
* adversarial validation
* correlation analysis
* time consistency
* client consistency  
* train/test distribution analysis
  
One interesting trick called "time consistency" is to train a single model using a single feature (or small group of features) on the first month of train dataset and predict `isFraud` for the last month of train dataset. This evaluates whether a feature by itself is consistent over time. 95% were but we found 5% of columns hurt our models. They had training AUC around 0.60 and validation AUC 0.40. In other words some features found patterns in the present that did not exist in the future. Of course the possible of interactions complicates things but we double checked every test with other tests.

# Validation Strategy
We never trusted a single validation strategy so we used lots of validation strategies. Train on first 4 months of train, skip a month, predict last month. We also did train 2, skip 2, predict 2. We did train 1 skip 4 predict 1. We reviewed LB scores (which is just train 6, skip 1, predict 1 and no less valid than other holdouts). We did a CV GroupKFold using month as the group. We also analyzed models by how well they classified known versus unknown clients using our script's UIDs.  
  
For example when training on the first 5 months and predicting the last month, we found that our 
* XGB model did best predicting known UIDs with AUC = 0.99723
* LGBM model did best predicting unknown UIDs with AUC = 0.92117
* CAT model did best predicting questionable UIDs with AUC = 0.98834

Questionable UIDs are transactions that our script could not confidently link to other transactions. When we ensembled and/or stacked our models we found that the resultant model excelled in all three categories. It could predict known, unknown, and questionable UIDs forward in time with great accuracy !!


[1]: https://www.kaggle.com/c/ieee-fraud-detection/discussion/111284
[2]: https://www.kaggle.com/kyakovlev/ieee-basic-fe-part-1
[3]: https://www.kaggle.com/kyakovlev/ieee-uid-detection-v6
[4]: https://www.kaggle.com/kyakovlev/ieee-basic-fe-part-1
[5]: https://www.kaggle.com/kyakovlev/ieee-fe-for-local-test
[6]: https://www.kaggle.com/alijs1/ieee-transaction-columns-reference
[7]: https://www.kaggle.com/cdeotte/eda-for-columns-v-and-id
[8]: https://www.kaggle.com/cdeotte/xgb-fraud-with-magic-0-9600
[9]: https://www.kaggle.com/c/ieee-fraud-detection/discussion/111510
