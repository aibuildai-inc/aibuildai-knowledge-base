# [42nd Place Solution] Ensemble + Classification

Competition: linking-writing-processes-to-writing-quality
Rank: #38
Source: https://www.kaggle.com/c/linking-writing-processes-to-writing-quality/discussion/466862

First of all, I’d like to express my gratitude to Kaggle and the host for organizing this great competition. It’s grateful to obtain our first silver medal, and to become a competition Expert!

I also want to extend my thanks to @hiarsl, @awqatak, @mcpenguin, @alexryzhkov, and many other competitors for sharing their wonderful accomplishments and insights. We learned a lot from you all. And last but not least, I thank my teammates @kensumida and @keisnoopy for their amazing contributions. Thanks!


# Solution
Here’s a brief illustration of our model.


## Ensemble Public Notebooks
Among high-scoring public notebooks, we referred the following.
・LGBM (X2) + NN | LB: 0.582 @cody11null
・Writing Quality(fusion_notebook) | LB: 0.580 @yunsuxiaozi
・LGBM (X2) + NN + Fusion | LB: 0.578 @kononenko
・VotingRegressor (7 models) + 165 Features [optuna] | LB: 0.580 @minhsienweng

We blended each single model by 0.1 * (LGBM + NN) + 0.2 * (Public LGBM) + 0.3 * (fusion_notebook) + 0.4 * (VotingRegressor).


## Classification of High-scoring Essays
Since the scoring metric was RMSE, a small portion of outliers could largely impact our score. Using the public LGBM model by @awqatak, we assessed how much edge targets (0.5 or 6.0 pts) contribute to worsen the score. As shown below, these targets had a larger influence on errors though they accounted for only 1.7 % of the train data. 

 
This was because the LGBM model was not super good at predicting those edge values, as its minimum and maximum predictions were 1.3 and 5.4, respectively. 

To deal with this, we separately trained a LGBM classification model to detect essays with 6.0 pts, and post-processed ensemble submission by substituting the scores if the probability of being 6.0 pts is high. The ROC-auc was ~0.90, which was high enough to boost our model performance. In our top selections, replacing the prediction values to 5.5 worked for us. 

This idea comes from discussions on the nature of RMSE and suggestions from @alexryzhkov that explained the possibility of applying classification. Thank you!
