# 6th place solution

Competition: child-mind-institute-problematic-internet-use
Rank: #6
Source: https://www.kaggle.com/c/child-mind-institute-problematic-internet-use/discussion/553146

Hello to all the competition participants. 
Thank you to the competition organizers for providing us with this valuable experience.

I gave up trying to catch up with the more skilled participants two months ago.  I am very surprised and confused by the change in ranking after I gave up.  I recognize this result is due to luck, not my ability.

I will share the solution that gave me such unexpected results below.

**Overview**
- Models
Three models ensemble (simple average).
The models are Vision Transformer (change the input layers code), lightgbm, catboost.
- Preprocessing
Aggregate parquet data into one row per id using mean, std, etc.
Convert categorical variables using 'to_dummies (polars)'. 
Impute null values ​​using 'group_by('Basic_Demos-Age', 'Basic_Demos-Sex').mean()' in the training data.
Standardize features using min_max of the training data.
- Learning
Metric : mae
Performe cross-validation using 'StratifiedKFold (n_splits=5, y:'sii')'.
Like many other participants, Use 'threshold_rounder' (after the ensemble).
- Notebook
[here](https://www.kaggle.com/code/miyafuru/internet-use-v2?scriptVersionId=201867541)

**Result**
　CV(before threshold_rounder) : 0.408,  CV(after threshold_rounder) : 0.481,  Public LB : 0.471,  Private LB : 0.476

**Changes that worked**
- Using the Transformer
Best result before use（twe models ensemble）
CV(before threshold_rounder) : 0.389,  CV(after threshold_rounder) : 0.483,  Public LB : 0.463,  Private LB : 0.472

**Changes that didn't work**
- Optimizing the ensemble weights
Best result
CV(before threshold_rounder) : 0.398,  CV(after threshold_rounder) : 0.483,  Public LB : 0.454,  Private LB : 0.463
- Metric : quadratic weighted kappa
Best result
CV(before threshold_rounder) : 0.429,  CV(after threshold_rounder) : 0.485,  Public LB : 0.457,  Private LB : 0.470
- Using 'PCIAT-PCIAT_Total' as target
Best result
CV(before threshold_rounder) : not calculated,  CV(after threshold_rounder) : 0.481,  Public LB : 0.461,  Private LB : 0.471

I hope this post helps you understand the surprising ranking changes.
Thank you for reading !
