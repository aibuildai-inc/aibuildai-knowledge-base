# 9th Place Solution

Competition: playground-series-s3e6
Rank: #9
Source: https://www.kaggle.com/c/playground-series-s3e6/discussion/389151

My brief write-up of the 9th place solution can be found in the attached link. 

https://www.kaggle.com/code/brendanmoore14/9th-place-solution

A simple automl approach using autogluon, with the original data included within the training loops, but not the cross-validation, ended up being my highest local cross-validation. 
- I used a 5-fold StratifiedKFold split, with the classes being the three groups of price/sqm (~10/sqm, ~100/sqm, and ~1000/sqm). 
- I tried multiclass classification on these three groups, but it ultimately did not improve the modelling. 
- Feature engineering, oversampling of the minority classes (~10/1000/sqm), and undersampling of the majority classes (~100/sqm) were also not helpful in my experimentation to improve the local cross-validation.  

In the notebook, I have outlined what worked for me and what did not. Unlike some of the other top scores, I was not able to achieve high local cross-validation using a single model.
