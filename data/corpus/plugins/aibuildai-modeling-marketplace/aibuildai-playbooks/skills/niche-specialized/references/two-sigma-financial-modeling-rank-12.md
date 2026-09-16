# #4 Public, #13 Private Solution

Competition: two-sigma-financial-modeling
Rank: #12
Source: https://www.kaggle.com/c/two-sigma-financial-modeling/discussion/29518

Our solution (Giba and Xpeuler) is a basic blend of 7 models:

 1. Ridge A - Selected Features trained on SignedExp( y+1 ). Some cleaning on features and filter on target instances.
 2. Ridge B - Selected features and some cleaning on features and filter on target instances.
 3. Ridge C - Selected features and some cleaning on features and filter on target instances. 
 4. Extra Trees - Selected features. 222 trees
 5. XGB - Selected Features and tuned hyperparameters on "all" trainset.
 6. Ridge online rolling fit - Trained every 100 steps on submission time. features: Some lags of [technical20-technical30].  Target used:  lag1 of [technical20-technical30] 
 7. Variance by Step(day) - Simple variance calculated over all 'Id' per day 

The final predictions are a weighted average of that 7 models.
 
Crossvalidation for model performance and feature selection was made using some approaches:

 - 2 folds:   timestamp > 906 and timestamp <= 906

 - 5 kfolds

 - rolling fit for ts> 906

Our solution is available at Kernels. Run time is about 35 minutes.  
https://www.kaggle.com/titericz/two-sigma-financial-modeling/team-rocket-13 

If you press "RUN" button you will get a very good overfitted score. It's because XGB was trained on all data.

Sorry for the lack of comments.

Take care, there is some tricks in the script that you can call "black magic" ;-P

Have fun...as we had!

obs. Don't forget to UP vote  ;-D
