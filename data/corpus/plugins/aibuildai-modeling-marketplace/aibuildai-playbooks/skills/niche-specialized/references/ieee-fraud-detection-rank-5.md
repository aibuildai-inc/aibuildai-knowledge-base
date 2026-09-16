# 5th Place Solution: Lions

Competition: ieee-fraud-detection
Rank: #5
Source: https://www.kaggle.com/c/ieee-fraud-detection/discussion/111735

First of all, I would like to thank Kaggle and IEEE-CIS and Vesta corporation for hosting this competition. Congratulations to the winners. Thanks to the best team members @hmdhmd, @shuheigoda, @yryrgogo who were able to share great results.

In this solution, we describe what we have devised, while omitting what has already been explained in other solutions as much as possible.

## Final submission summary
* Simple average of LightGBM individually modeled by four team members. The effect of averaging was great even though everyone modeled using LightGBM &amp; similar parameters.
  - Although our team was formed from the beginning, we almost didn’t share modeling ideas except the core part of the competition (e.g. userID). This is one of the factors that greatly improved the ensemble.
* After averaging, we override isFraud to 1 as following two user patterns. This gave us about `+0.0011` boost in private LB.
    1. Users who are certain to exist in train dataset and isFraud=1. (335rows)
    2. Users who are confirmed as isFraud=1 in test-public dataset by LB probing. (207rows)
* stats
    - Final Result: `private 0.942453 / public 0.964383`
    - Base Models (LightGBM)
        - private 0.933088 / public 0.956396
        - private 0.938111 / public 0.960937
        - private 0.936381 / public 0.959820
        - private 0.933819 / public 0.958436

## Key Ideas
### User identification
- User identification was a very important factor, but we briefly describe as it has already been explained in many other solutions.
- Column used for identification: `D1/D3/addr1/P_emaildomain/ProductID/C13`
- The identification work was performed with attention to the following two points.
    1. emaildomain: Replace `anonymous.com` and `mail.com` with `NaN`
    2. There is a gap period between train and test-public
        - Recognized as the same user even if `D3` etc. don't fit due to the gap period
- In the above user identification is not perfect, so we aggregate based on various keys to generate user group features as describe details below.

### User Group Feature
- This is the same type of feature that is called userID , but in one model we didn't try to identify the User. The concept of the model is to let the tree model specify a userID . We added a lot of necessary features to the model.
- We made about 80 features like this `df.groupby('uid_and_register_date')[C_or_V_columns].agg(['mean'])`. This is similar to 1st place solution as chris described. Single LGB improvement by these 80 feature is following: `LB 0.9519-&gt;0.9565, CV 0.9257-&gt;0.9339`
- Since this user grouping is not perfect neither, we have created features to identify different users in user groups.
    - Whether the value matches the most frequent value in the group
    - Whether the value appeared in the user group last month
    - How many days ago that value appeared in the user group
    etc...

### Utilizing test data which we grasp isFraud correctly
- Since we were able to grasp isFraud of more than 5000 lines(1) of test data almost accurately by user identification and probing, we utilized as following three methods. (1: total 5448rows / positive: 542 / negative: 4906))
    1. Override at final submission (as wrote above)
        - Since the loss is very large when the one with true value is overridden as 0, we only override the one with true value isFraud=1.
        - This gave us about `+0.0011` boost. (Even if we didn't override, we got the gold medal.)
    2. Feature Selection
        - We calculated AUC with the data in the test set for which we knew the correct answer. If we add new features to model and the AUC dropped significantly, we threw away them.
    3. Determination of model ensemble ratio
        - The ensemble ratio was decided to maximize the value of above AUC calculated only in the test-private. (I'm not sure how useful this was.)
        - Weight optimize was executed using gpyopt.
- The isFraud specification could be used to identify negative. To be specific, if user who is isFraud=0 in train and has completed all transactions within 120 days from the last transaction date of train, we can judge all(train&amp;test) transactions as isFraud=0.

### Feture Selection Policy
- At the beginning of the competition, features that contributed to Adversarial Validation were not added because of fear of shakedown due to the use of features with large discrepancy in train / test.
- However, from the experiment described below, it was found that even a feature with a large discrepancy in train-test contributes to improving the score. Therefore, we changed the policy to add features that contribute to improving prediction accuracy, even if they contribute to Adversarial Validation.
- Experiment
    - Divide train data into 3 parts by the following ratio
        - (Pseudo) train: 50%
        - (Pseudo) public: 10%
        - (Pseudo) private: 40％
    - We investigated the relationship between adv val AUC and oof, pub, priv AUCs under different conditions.
    - Regardless of the value of AUC of Adversarial Validation, if oof is high, pubic and private AUC tend to be high

## Model details
* Four team members used LightGBM with almost the same settings
    - validation: `DT_M` GroupKFold
    - userID GroupKFold didn't work well.
    - parameter: almost same as konstantin's kernel.
* I(ML_Bear) made original model using only the data of users who have only one transaction.
    - purpose: If a user with a large number of transactions is in a scam, normal transactions will be overridden to fraud, making it really difficult to see what the scam transaction is. Therefore, I tried to extract the characteristics of isFraud using transactions of only one transaction users.
    - validation: StratifiedKFold
    - The oof / pred of this model was input to above LightGBM as feature.
    - parameter: almost same as konstantin's kernel.
    - Thanks to this special model(?), the public LB achieved by my model was not very high, but the weight of the ensemble was fairly high. (best my model: `private 0.935089 / public 0.958713`)

## Feature Engineering
### Based on userID
We used the userID excessively.
- `user_transaction_per_day`
- `user_transaction_amt_min`
- Aggregate the average and deviation of `TransactionDT.diff(1)` for the last N (= 2,3,4,5,10,20) rows.
- Counting the number of nulls for each userID for columns that contain 1% or more of nan.
- Aggregate the deviation, max, min, diff of max and min, and unique count (or rate) for `Cxx` or `Dxx` or 'V_group' fetures. (`userID_Vxx_Vxx_mean_mean`, `userID_Cxx_unique_rate`, etc.)

### Other features
- Aggregation such as `TransactionAmt` with various keys including `regist_date` (Transaction date - D1)
- `Cxx` feature dimension reduction by PCA
- TargetEncoding of `addr1` (Numerical features but TargetEncoding worked)
- category groupby feature
    - Calculate frequency distribution of specified category data for each groupby key
    - Expresses the probability of fraud when it behaves differently from the usual by the probability of appearance of the category
- LDA / SVD of category co-occurrence matrix
- `Vxx` encodings
    - Since `Vxx` has many features, we tried a few encodings instead of using original values.
    - For example, since `V242~258` group has almost the same value scale, count encoding was performed for each value in that group. ("1" count in V242~258, "2" count in V242~258...)

#### In single transaction users model
- TargetEncoding of `TransactionAmt` (Numerical features but TargetEncoding worked)
- TargetEncoding of `emaildomain` (both P and R)
- `TransactionAmt_mod_XXX_is_zero` (XXX=50,100,200)

#### Other trivial content
* START_DATE
    - We set `transactionDT=0` as `11/30 00:00:00`
    - Estimate approximate date from browser history
    - Looking at the number of transactions, we saw a pattern of "many transactions for 5 consecutive days → fewer transactions for 2 consecutive days". So we estimate that the day with few transactions is a Saturday or Sunday, then 11/30 fits perfectly.

## What did not work
- Seed Averaging
    - There was no high effect.
    - Averaging a model with high AUC in test data and a model with high AUC in train oof worked a little
- Split model for each `ProductCDs`.
    - We thought that the Fraud trend in `Cxx` and `Vxx` was different depending on `ProductCD`.
    - There were some features created for each `ProductCD` that helped improve CV &amp; LB, but not enough.
- Change sample_weight according to the number of transactions
