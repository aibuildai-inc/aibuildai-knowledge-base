# 10th place solution

Competition: elo-merchant-category-recommendation
Rank: #10
Source: https://www.kaggle.com/c/elo-merchant-category-recommendation/discussion/82093

Thanks to all competitors for this wonderfull day) I am very happy to get first Gold and become Kaggle Master)

**PREPROCESSING and FE**

 1. Merging historical data and new_data with authorized_flag=1
 2. Count Vectorizer merchant id, subsector id, merchant category id + PCA, SVD
 3. Grouping month lag data (auth month-13, auth month-12 and etc.) for count purchases, and sum adjusted purchase_amount, std, min, max
 4. Ratio every auth_month_purchase_amount-13,...auth month purchase amount+2 to previous with interval 2,4,6
 5. Predicting month lag 3 and 4 and calculating ratio for them
 6. Simple Rolling mean and SimpleExpSmoothing over auth purchase amount-13, ...auth purchase amount+4 over month lags
 7. Calculating avg, min, max purchase time elapsed periods beetween
    transactions

**FEATURE SELECTION**

Finally, i created almost 6500 features and created different features sets using

 1. Boruta (it took almost 8 hours to select 500 best features)
 2. Selecting best features based on Catboost, LGBM, XGB
 3. Deleting features based on Adversarial validation

**Best Single Model and Ensembling**

My best single model is Catboost over 5 stratified folds CV 3.645 std 0.012 LB 3.671. Also tried DeepFM but CV 3.656 std 0.016 and LB 3.682. Other models, including LGBM sometimes got even better CV score, but were worse on LB. 

**Finally, i collected 47 models with outliers, and 35 models without them**. For combining submissions, also created classification model for outliers ROC 0.907 F1 0.1025

 1. Stacking models with outliers CV 3.637 std 0.013 LB 3.669
 2. Stacking models without outliers CV 1.544 std 0.011
 3. Combining models, replacing 10000 values

**Thanks for sharing ideas end experience**

[@raddar][1] Post about [real meaning of target ][2]  and [@peterhurford][3] for this [feature selection post][4]


  [1]: https://www.kaggle.com/raddar
  [2]: https://www.kaggle.com/raddar/target-true-meaning-revealed/
  [3]: https://www.kaggle.com/peterhurford
  [4]: https://www.kaggle.com/c/elo-merchant-category-recommendation/discussion/73937
