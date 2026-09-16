# 9th place solution (nagiss part)

Competition: santander-customer-transaction-prediction
Rank: #9
Source: https://www.kaggle.com/c/santander-customer-transaction-prediction/discussion/89302#latest-516440

In my case, the key to improve the score was modeling independency (no interaction) between variables.

I made two types of models with such property.

## model 1 - 2step lgbm (LB 0.922)
The summary of the model is as follows.
[2step_lgbm_model]
Kernel: https://www.kaggle.com/nagiss/9-solution-nagiss-part-1-2-2step-lgbm

`num_leaves=2` means the model considering no interaction.

## model 2 - weight sharing NN (LB 0.923)
[weight_sharing_nn]
Kernel: https://www.kaggle.com/nagiss/9-solution-nagiss-part-2-2-weight-sharing-nn
If using NN, we can control interaction directly.
(Actually, weight sharing was caused by accident, but it works very well.)
