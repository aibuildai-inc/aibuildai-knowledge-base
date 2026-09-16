# 29th Place Solution

Competition: equity-post-HCT-survival-predictions
Rank: #29
Source: https://www.kaggle.com/c/equity-post-HCT-survival-predictions/discussion/566685

Here's a simple solution that gives you a 29th place.

## Stage 1
1) Estimate probability of EFS by fitting a CatBoostClassifier.
2) Estimate EFS_TIME for the subsample where EFS==1 using CatBoostRegressor.
Estimates in 1) and 2) are obtained OOF using 10 folds.

## Stage 2
Combine the estimates in Stage 1 to create a metric that maximizes ~~ Concordance Index~~ (EDIT: not CI but Competition Metric). One simple approach, which gives ~~CI~~ Competition Metric of around 0.69, is to take the ratio of EFS probability and EFS_TIME . A more elaborate approach which gives you a 29th place is to use Optuna to find the optimal coefficients $$a,b,c,d,e,f$$ that maximize the ~~CI ~~ Competition Metric of the following ratio: 

$$ \frac{a\*efs^3 + b\*efs^2 + c\*efs + 0.001}{d\*efs\\_time^3 + e\*efs\\_time^2 + f\*efs\\_time + 0.001}$$

The final solution uses the ensemble of 10 models from Stage 1 with their corresponding coefficients obtained in Stage 2.
