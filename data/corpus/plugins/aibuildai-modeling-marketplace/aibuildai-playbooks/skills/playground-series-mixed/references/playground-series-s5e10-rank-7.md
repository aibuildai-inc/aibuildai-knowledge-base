# 7th Place - Ridge

Competition: playground-series-s5e10
Rank: #7
Source: https://www.kaggle.com/c/playground-series-s5e10/writeups/7th-place-ridge

Thanks everyone for the fun competitions! Special thanks to:

* @masayakawamata – for sharing the NN stacking and TabM notebooks,
* @mikhailnaumov – for sharing the YDF model notebook,
* @metamodels – for the Bayesian + Residual LightGBM notebook.

**Models:**

* XGBOOST – 9 K-Fold with Target Encoding (5 K-Folds, smoothing = 10, aggregation = mean) and Counter Encoding, using all pairwise feature combinations.
* NN – based on @masayakawamata’s notebooks.
* LGB – hyperparameter tuning based on @metamodels’ notebook.
* XGB – 7 Stratified K-Fold + Feature Engineering.
* XGB – 5 Stratified K-Fold + Feature Engineering.
* YDF – based on @mikhailnaumov’s notebook.
* TabM – based on @masayakawamata’s notebook.
* XGB + LGB + CAT


**Stacking:**

* I used Ridge Regression as the meta-model with 15 K-Fold cross-validation.
