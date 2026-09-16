# 16th Place Solution "Magic Trick"

Competition: playground-series-s3e19
Rank: #16
Source: https://www.kaggle.com/c/playground-series-s3e19/discussion/428438

Hello to everybody!

First of all, I want to thank Kaggle for organising this fantastic contest. However, the Playground series is an excellent location to learn and advance in machine learning and data science. Many new participants find it difficult to deal with the enormous volumes of complex data in many highlighted competitions.

Since this was my first best rank in time series forecasting, this competition really benefited me.
I want to share what I've learned through this competition with others by sharing my solution.

Mostly I referd top score public notbooks to get the idea about feature engineering and molding, I Traind three model but non of the get  SMAPE < 7.44. Model I used Below Listed-
1. LGBM Regressor
2. Cat Boost Regressor
3. XGB Regressor

In the end of the week i try ensombling the Different  models: Thanks to @paddykb and @christph For there works that help me lot to boast my public LB.
https://www.kaggle.com/code/christph/gam-with-holidays
https://www.kaggle.com/code/paddykb/ps-s3e19-tableau-eda-gam-fit 

**So I decide to mean ensemble total 5 models and got 6.58 on LB and 7.44 on private LB** 

## Magic Trick
In the last day of deadline i try some magic trick. Idea comes from @ravi20076 [discussion about submission](https://www.kaggle.com/competitions/playground-series-s3e19/discussion/425973). As he suggested rounding the prediction increasing the LB **so i decide to round my submission and got improvement on LB 6.58  to 5.800.**

**After looking the improvement only rounding the prediction i got idea to add 1 in every prediction may increase LB and it does my LB improved 5.800 to 5.596 on LB and On private 6.47 and score 16th place.**

Thanks 
Happy Learning!
