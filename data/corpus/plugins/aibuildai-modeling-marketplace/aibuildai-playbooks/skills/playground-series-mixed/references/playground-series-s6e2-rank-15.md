# 15th Place Solution

Competition: playground-series-s6e2
Rank: #15
Source: https://www.kaggle.com/c/playground-series-s6e2/writeups/15th-place-solution

This was my first Kaggle competition that I spent a significant amount of time on. Along the way, I learned a lot from public notebooks and from previous winners of the Playground competitions. I wanted to try more ideas, but due to work commitments I couldn’t experiment as much as I hoped.

My solution was fairly simple compared to many others. I adopted @cdeotte’s nested K-fold approach from previous competitions and trained XGBoost and CatBoost models with a hillclimb ensemble (Public LB: 0.95390, Private LB: 0.95534). I also tried adding logistic regression, but it hurt performance, so I left it out..

[Solution link](https://github.com/EmreDuman221/Predicting-Heart-Disease---S6E2)  

**Overview**

This solution is a tree-based + boosting pipeline with careful leakage control for encodings and a simple but effective per-fold hillclimb blend between XGBoost and CatBoost. Performance comes mainly from:
- Using the provided original dataset as extra labeled training data
- Adding frequency encodings and numeric-as-categorical representations
- Performing out-of-fold target encoding (inner CV) to avoid leakage
- Blending XGB + CatBoost per fold with a small hillclimb search on AUC

Final submission is the average of blended predictions across folds.

**Feature engineering**
**Base feature groups**

Categorical (CATS): `Age, Sex, Chest pain type, FBS over 120, Exercise angina, Thallium`

Numeric (NUMS): `BP, Cholesterol, Max HR, ST depression, Slope of ST, Number of vessels fluro, EKG results`

Note: some “NUMS” are effectively discrete/ordinal; treating them as categorical helped.

**Frequency encoding (global)**

For each numeric/discrete column in NUMS, I computed a frequency (`value_counts` normalized) using train + original + test combined to stabilize rare values:

`FREQ_col = P(value)` for each value in the column

This creates new numeric features like:

`FREQ_BP, FREQ_Cholesterol, …`

**Numeric-as-categorical**

For each column in NUMS, I created:

CAT_col = col.astype(str).astype('category')

This allows the models (especially CatBoost) to capture non-linear level effects and interactions in discrete numeric-like variables.

**Feature interactions**

To enrich signal without exploding feature space, I computed correlations among base X columns and selected the top 8 absolute-correlation pairs, then added multiplicative interactions:

**a_x_b = a * b**

**Validation strategy**

I used nested 15-fold KFold on the competition training set.

Within each outer fold:

Training set = outer-train split + full original dataset

Validation set = outer-val split from competition data only

**Leakage-safe target encoding (inner CV)**

Inside each outer fold, I ran an inner 15-fold KFold on the outer training data and created OOF target encoding features for:

`TE_COLUMNS = (NUM_AS_CAT + CATS)`

statistic: mean

Process (per outer fold):

Initialize TE1_col_mean as NaN.

For each inner split:

compute mean target per category using inner-train

apply to inner-val rows and write into TE column

Fill remaining NaNs with global mean.

Fit final TE mapping on full outer-train and apply to outer-val and test.

After TE creation, I drop the original categorical columns used for encoding (so the model sees TE features rather than raw categories for those columns).

**Models**

I used XGBoost and CatBoost, I also added logistic regression, but did not improve my solution. I will try to diversify my models in the upcoming competitions.

**Ensembling: per-fold hillclimb blend**

For each outer fold, after training both models:
- `maximize AUC( w*val_xgb + (1-w)*val_cb )`

Then apply the same weight to that fold’s test predictions and accumulate.

Final prediction:

- average of blended test preds across folds
