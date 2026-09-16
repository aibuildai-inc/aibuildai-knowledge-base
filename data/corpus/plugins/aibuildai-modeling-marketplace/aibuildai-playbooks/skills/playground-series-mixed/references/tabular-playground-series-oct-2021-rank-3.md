# 3rd place solution

Competition: tabular-playground-series-oct-2021
Rank: #3
Source: https://www.kaggle.com/c/tabular-playground-series-oct-2021/discussion/284594

Hello every one,

My solution is a composition of public and private autoML solutions whose gradient boosting models (XGB, LGB, CAT, HGB) take an important place.
As @olivrk mentioned in his solution, the little extra to perform high in this competition is to combine one or more efficient NN models.
I mainly used https://www.kaggle.com/kavehshahhosseini/tps-oct-2021-multi-input-neural-network with 25 seed iterations in order to make the results **robust**.
Finally a last step consisted in retraining the main models by including pseudo-labeling (if proba test <0.05 => then target = 0 if proba test> = 0.95 => 1 and add test lines in train) to add an extra performance 0.000X.

Shall we meet for the November competition ? same AUC metric.
Small hint: the NN models perform much better in this new competition

Best to you,
Mathurin
