# 20th place solution 😂

Competition: data-science-bowl-2019
Rank: #20
Source: https://www.kaggle.com/c/data-science-bowl-2019/discussion/127650

Congratulations to all winners!
Here is brief summary of our solution.

### Feature Engineering
- worked
   - is 1st assessment or not
   - Normalized counting feature: count for each event codes and ids / game session duration
   - Whether assessment is solved in order of the game design, or not
   - aggregations
      - durations for each type (mean, std, min)
      - the number of records for each type (mean, std, min)
- not worked
   - last activity, last game statistics
      - corrects, incorrects, misses, rounds, levels..
   - last type, title history sequence

### Models
Using QWK for tuning models was too difficult, so we decided to evaluate only the RMSE for the model performance.
Group 5 folds is used as validation method. We applied truncation to validation set.
- 1st level
    - lgbm: CV 1.0395 +/- 0.031
      - objective rmse
      - 3 random seed averaging
    - xgb: CV 1.0457 +/- 0.028
      - objective rmse
      - 3 random seed averaging
    - catboost: CV 1.0430 +/- 0.028
      - objective rmse
    - NN: CV 1.0423 +/- 0.029
      - rmse + smooth l1 loss
      - RNN-layer: GRU + Attention
         - sequence of last 6 histories as input
      - Dense-layer
      - 3 random seed blending
      - NN model has the almost same performance as boosting tree models, but has a low correlation.
- 2nd level: PublicLB 0.538 PrivateLB 0.556
    - ElasticNet: CV 1.0361 +/- 0.028
In addition, we cloud not include it though, lgbm with accuracy classification is best model in our experiments.

### Thresholding
The most time was spent on how to determine the threshold. We prepared some ides and experimented.
We did sampling with replacement 10 times from each fold's OOF for each installation_id (we called it as OTV).
Then, we applied following thresholding methods with 5 folds CV of OOF.
- 1. Confirm label distribution of validation set to OOF, OTV true label distribution (many kernel did)
- 2. Apply Optimized Rounder to oof, otv and get thresholds from them, then confirm the validation label distribution to optimized OOF, OTV label distribution.
- 3. Apply Optimized Rounder to OOF, OTV and get thresholds from them, then apply thresholds to validation set.
In our experiments, No. 2 method with otv almost got top score, but sometimes No.3 with OOF did.
So we chose two thresholding methods as final submissions.
Experiment results close to PrivateLB.
The results of the above experiment gave almost the same results as PrivateLB.

[results]

### train dataset augmentation
The extension of train set by test set was used only for training each 5 folds, and was not used for determining the thresholds or as validation set.
