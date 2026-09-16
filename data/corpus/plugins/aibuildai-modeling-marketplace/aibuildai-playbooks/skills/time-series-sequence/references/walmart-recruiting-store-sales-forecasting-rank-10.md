# Thank You and #2 rank model

Competition: walmart-recruiting-store-sales-forecasting
Rank: #10
Source: https://www.kaggle.com/c/walmart-recruiting-store-sales-forecasting/discussion/8023#43854

<p>My approach is an arithmetic mean of 3 models:</p>
<p>- First I translated all datasets using a spline to show day-by-day values.</p>
<p>- Model 1: Custom linear regression optimizing MAE, using only 2 features (naive observation of previous&nbsp;2 years). public LB ~ 2490</p>

<p>- Model 2: GLM, features: last year observation + features from Features.csv. Public LB ~ 2650</p>
<p>- Model 3: GBM, features: last year observation + features from Features.csv. Public LB ~ 2700</p>
<p>The right choose of the cross-validation periods is very important.</p>
