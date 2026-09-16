# 6th Place Solution - Lightgbm with Target flattened

Competition: godaddy-microbusiness-density-forecasting
Rank: #6
Source: https://www.kaggle.com/c/godaddy-microbusiness-density-forecasting/discussion/417821

Great Thanks to kaggle and Godaddy to hold this competition. Congratulations to all the winners. And many thanks go to people who created some mazing public notebooks.
Here is my Solution:
# Validation
- Last 12 month data for validation. 
- To evaluate any solution is validate, I additionally compute the number of improve months of the 12 valid month instead of simple average cv improve.Because some month have abnormal high/low raising rate:`mbd(curr_month+1)/mbd(curr_month)-1`,average cv may lead to overfit


#  Outlier smooth
- Mean smoothing for 2022-06->2022-08 to handle the general sharp rise-fall phenomenon 

 - There are also other data points have the phenomenon like "origin->Sharp rise-> Sharp fall->origin".But I haven't found the best way to deal with them yet.
- Basic constant smoothing by fraction from giba
- Stable blacklist for those nearly unchanged cfips
 - [28055, 13101, 13265, 31009, 31115, 31149, 38047, 38087, 48033, 48301]
# features：
  - Action diff, Target shift, Target diff
  - Target diff window 
     - Sum
     - Std
  - Target window
     - Sum
     - Quantile 0.2
     - Quantile 0.8
     - Lowwer bound：
         - only have value when sign(Quantile 0.2)=sign(Quantile 0.8)，0 for other situations
         - Quantile 0.8 for negative value
         - Quantile 0.2 for positive value
  - State cluster
# Enhance modeling framework
construct the Enhance model frame, input 1)feature and 2) n_cross: number of future month to predict, output the raising rate of mbd in future n_cross month. In this way we can directly predict the public/private mbd by changing n_cross from 1-5
Detail structure:
- Enhance modeling frame features=features+n_corss
- n_corss=num of month to predict future mbd
- output target：[mbd(curr_month+n_corss)/mbd(curr_month)]**(1/n_corss)-1
- Model：Xgboost 
# Target_flatten：Train with flattened target, and do reverse transform in prediction
- The distribution of target variables changes with the change of active base. The lower the active base is, the higher the absolute value of the target is.

- Do a simulate to transform all active base into same target distribution. We assume the distribution will be flatttened after the transformation, in this way we can construct the transform formula by testing:
  - `coef=(1/(0.007*(active_series/10+105))+1)`
  - `target_flatten=target_series/coef`
- Train with flattened target, and do reverse transform in prediction
# Post process
- I noticed that the trend of public leaderboard month is abnormal higher than normal months.This phenomenon also happends on other month like 2022-07, In my opnion, such whole month trend is unpredictable, our model should not overfit it.
-  Considering that the mbd for the public month is the basis for the subsequent private month, I post-processed the forecast results like:
`prediction=prediction*(1+0.001)`
