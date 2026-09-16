# 35th Place Solution: Ensemble 10 Models with Hill Climbing

Competition: equity-post-HCT-survival-predictions
Rank: #35
Source: https://www.kaggle.com/c/equity-post-HCT-survival-predictions/discussion/566675

Very happy to get my first silver medal, and this was also the first time I attended a competition from the beginning to the end without giving up halfway. 
# Solution Overview
My solution consists of three parts: 
Part 1.  Models derived from Chris's notebooks. I use XGBoost, LightGBM, and CatBoost with KM, NA, and Cox target transformation. Furthermore, each model is trained on two  random seeds and then averaged. 
Part 2. LitNN with pairwise ranking loss. This model can boost my CV and LB significantly, so I fine-tune them on three random seeds. One of these models can reach a CV of 0.683, but it always harms my ensemble's CV, so I don't include it in my final submission. 
Part 3. Yunbase. I use the public yunbase without changing anything. This model increases my CV from 0.6869 to 0.6876. I think its unique feature engineering and training settings boost diversity in my ensemble.
Finally, I use a hill climbing algorithm to ensemble these models with the search step of 0.01. In my case, hill climbing is better than Optuna.

# Useful Tricks
1. **Power transformation (scipy.stats.yeojohnson)**. In Part 1, I apply power transformation to the efs_time before any other target transformation. This thought comes from Chris, because RMSE and MSE Loss are suitable to the normalized target. I employ it separately in the event/survival group.
2. **Enable negative weights in hill climbing**. I find that averaging Part 1 and Part 2 models directly leads to the degradation of CV and LB. I guess that's because some models are biased (tend to predict higher/lower than the ground truth). Therefore, I enable negative weights when ensembling, which significantly improve my CV and LB.

# Model Importance
This graph shows how the model's standardized out-of-fold prediction is added to the ensemble. Interestingly, many models with low CV are added to the ensemble first:
 

# Thanks
More importantly, I cannot reach this place without excellent public notebooks and enlightening discussions, I really learnt a lot of models and tricks in this competition. Thank you for your self-giving sharing!!
