# 34th Place. CatBoost with MultiRMSE plus public solutions.

Competition: equity-post-HCT-survival-predictions
Rank: #34
Source: https://www.kaggle.com/c/equity-post-HCT-survival-predictions/discussion/566858

My final solution/ensemble consisted of 4 models. 3 of them are based on public notebooks (maybe with minor modifications): 
1. AutoGluon on KaplanMeier target
1. Event Masked PRL-NN
1. XGBoost with Monotonicity

4th model is CatBoost with MultiRMSE loss and metric functions trained on 3 targets. Also using sample_weight with 1.2 for efs==1 and 0.5 for efs==0 allowed to improve model performance.
The weights for the final ensemble were found by scipy.minimize function that had the best CV and the best private LB.
So the solution is relatively simple, but it's allowed to take good enough position. Of course, making more efforts on modeling and ensembling would get better position and maybe on the gold zone.
