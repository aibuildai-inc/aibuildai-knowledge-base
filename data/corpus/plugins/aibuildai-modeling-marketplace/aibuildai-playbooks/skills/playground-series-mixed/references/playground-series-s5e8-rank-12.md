# 12th Place Solution - Stacking with XGB and NN

Competition: playground-series-s5e8
Rank: #12
Source: https://www.kaggle.com/c/playground-series-s5e8/writeups/12th-place-solution-stacking-with-xgb-and-nn

Congratulations to all participants! My solution for 12th place consists of:

* 28 level-1 models, most of which variations of GBDT and NN. Best single model was an LGBM with CV 0.9761 (5 folds).

* Stacking models XGB and simple NN (hidden layers 32, 32, 16) trained on the level-1 models. I found in this competition that non-linear stackers worked a lot better than linear methods like Ridge. NN stacker individually achieves 14th place on the private leaderboard and XGB achieves 18th.

* Simple average of the XGB and NN achieves 12th on private leaderboard.



Thanks to the creators of the following public notebooks which influenced my solution.
* @yekenot - https://www.kaggle.com/code/yekenot/ps-s5-e8-deeptables-nn
* @mahoganybuttstrings - https://www.kaggle.com/code/mahoganybuttstrings/pg-s5e8-single-xgb-cv-0-975782-lb-0-97681
* @cdeotte - https://www.kaggle.com/code/cdeotte/nn-by-gpt5-cv-0-974-wow
* @cdeotte - https://www.kaggle.com/code/cdeotte/xgboost-using-original-data-cv-0-976
*  @yunsuxiaozi - https://www.kaggle.com/code/yunsuxiaozi/pss5e8-tabm-baselinecv-0-9738
