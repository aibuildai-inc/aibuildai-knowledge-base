# #2 LB Approach

Competition: tabular-playground-series-feb-2021
Rank: #2
Source: https://www.kaggle.com/c/tabular-playground-series-feb-2021/discussion/222762

Unfortunately I didn't really manage to improve on my approach from last month. So for anyone interested please just reference January post & notebooks.
https://www.kaggle.com/c/tabular-playground-series-jan-2021/discussion/216071

Generally just changed some parameters etc, and dealt with new categorical features in the usual way.

Unfortunately didn't have much time in first half of the month to explore the DAE idea, then did not have much success in second half of month with making it work.

Congrats to @ryanzhang on first place by a wide margin also appreciate all the people sharing great ideas (e.g. fine tuning of LGBM model) during competition 👍

Edit: just as a quick note, the scores of individual models. Format Private/Public Score. All had CV scores pretty similar to public leaderboard.

LGBM 0.84282 / 0.84229
XGB 0.84356 / 0.84301
NN1 0.84362 / 0.84250
NN2 0.84333 / 0.84261

Blend weights calculated by optuna based on using CV OOF predictions:

params_lgbm_weight     0.468606
params_xgb_weight      0.082158
params_nn1_weight      0.197289
params_nn2_weight      0.251397

Output score 0.84228 / 0.84157
