# 32nd Place Solution

Competition: equity-post-HCT-survival-predictions
Rank: #32
Source: https://www.kaggle.com/c/equity-post-HCT-survival-predictions/discussion/567112

First of all, I'd like to thank the organizers and Kaggle for hosting such an amazing competition. I would also like to express my special gratitude to my teammates [@cody11null](https://www.kaggle.com/cody11null) and [stefanoclss](https://www.kaggle.com/stefanoclss).

# **Overview**
To improve the accuracy of the target transformation, we first transformed the survival times of each race_group to follow a log-logistic distribution, and then performed final model prediction on an ensemble of multiple GBDT+TabM and nn_pairwise models using the two types of target transformation.



# **Target Transformation**
Assuming that efs_time with long survival time (efs=0) and efs_time with short survival time (efs=1) follow different statistical distributions, found that the efs=1 data follow a unique log-logistic distribution by race_group. Then extracted the data with efs=0 that followed this log-logistic distribution, and corrected these data by the conditional expectation of this distribution. Using these data, two target transformations (Kaplan-Meier and QuantileTransform) were performed.

# **Training Models**
We performed some feature engineering and built the following model.
- Feature Engineering: One-hot encoding, Lavel encoding and custom features
- Model: 19GBDTs+7nn models as following; GBDTs (CAT, LGB, XGB) and TabM on KaplanMeier target, GBDTs (CAT, LGB, XGB) with Monotonicity, Event Masked PRL-NN, Yunbase Model etc
         
# **Ensamble model**
Each single models were divided into several blocks, and a stacking ensemble using the oof of each model was performed for each block in the first layer. A model was constructed that maximized CV and LB by increasing diversity for each block through repeated trials. In the second layer, a linear model was constructed in which the first layer ensemble was weighted to maximize the c-index of CV.Finally,post-processing was performed to balance the scores for each race_group and maximize the c-index.

# **Submission**
Below is our final submission result.
Best : CV 0.6875 / public LB 0.694 / private LB 0.694
