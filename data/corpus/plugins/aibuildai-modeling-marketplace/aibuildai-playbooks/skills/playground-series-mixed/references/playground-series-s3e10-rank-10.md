# Top 10 Solution

Competition: playground-series-s3e10
Rank: #10
Source: https://www.kaggle.com/c/playground-series-s3e10/discussion/396263

Congratulations to the winners of this competition! It was a really tight leaderboard.

My solution:
**Dataset**:
I trained only on the synthetic data.

**CV:**
15 Fold stratified. 

**RFE:**
I have not removed any features.

**RFA:**
I have added the following features which worked -
['EK_diff_Mean_DMSNR_Curve', 'Mean_DMSNR_Curve_diff_EK',
  'Mean_DMSNR_Curve_mul_EK_DMSNR_Curve',
  'Mean_DMSNR_Curve_mul_Skewness_DMSNR_Curve',
  'Mean_DMSNR_Curve_div_SD_DMSNR_Curve', 'SD_DMSNR_Curve_div_Mean_DMSNR_Curve'],

I did that using 2 models separately and added only features that improved both models.

**Preprocessing:**
I removed the observations based on the @dmitryuarov proposal [here](https://www.kaggle.com/competitions/playground-series-s3e10/discussion/393093)

**Ensemble:**
7 Ensembled XGB models across different seeds of Stratified KFold with different xgb params each time.

**What did not work:**

LGBM, NN, RF, RF with Global Refinement.

**What could be done differently:**
I did not trust the sub with @paddykb ensemble. It was giving 0.0307 private, 03101 public (straight top2).
