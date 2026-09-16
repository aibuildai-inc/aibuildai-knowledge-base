# #7 Private #3 Public approach| Simple ensemble with post-processing

Competition: playground-series-s3e14
Rank: #7
Source: https://www.kaggle.com/c/playground-series-s3e14/discussion/410666

Hello all,

I wish to extend sincere thanks to the Kaggle team for the episode and the playground series in totality. I think these playground series competitions are incredibly rewarding, offering lots of opportunities to beginners and practitioners alike. Firstly, I wish to congratulate the winners of the challenge and extend sincere thanks to all the participants for making this a memorable experience. I shall also extend sincere thanks to the below users in particular for their contributions over the past fortnight-
a. @paddykb 
b. @adaubas
c. @mattop 
d. @tetsutani 

**Feature engineering-**

1. I used only the features provided in the dataset without creating additional features. I used PCA and PLS to create association based features as mentioned in the below link-
https://www.kaggle.com/code/adaubas/ps-s3e14-stacking-leastabsolutedeviation-reg
2. I did not use any scaling/ centering options
3. I adjusted some data values in certain features (RainingDays and MaxOfUpperTRange) as mentioned in the below link, thanks to @paddykb and @adubas for the idea-  
https://www.kaggle.com/code/adaubas/ps-s3e14-stacking-leastabsolutedeviation-reg
4. I did not ensue any transformation on the target and used it as-is
5. I included the original data in my training, it helped me improve my CV score drastically compared to its non-usage. I created the OOF score purely from the competition data (as per my public notebook).

**Base Models-**
1. I initiated the challenge with lots of common methods, most of which did not work. I used **Repeated K-Fold** - 10x3/ 10x2 for all my models. 
2. I used optuna and FLAML for the model training and tuning, taking cues from the link below, thanks to @paddykb-
https://www.kaggle.com/code/paddykb/ps-s3e14-flaml-bfi-be-bop-a-blueberry-do-dah
I tuned very basic parameters on optuna with emphasis on the below parameters-
a. max depth
b. learning rate
c. reg-alpha and reg-lambda
d. number of leaves
I also resorted to manual perturbation of these parameters to ensure I do not end up overfitting to the noise in the data. 
3. I used the original data to build the model training folds, while the evaluation (dev-set) was the corresponding competition fold. Using the original data in this manner improved my CV score and my public leaderboard position quite well
4. I resorted to feature subsets across multiple models, using **fruitmass, fruitset and seeds** in all my models. I then used 1-3 features from the remaining categorical features in my base models. In particular, I used 1 temperature range column and 0-2 columns from the remaining category features to complete the feature subset for base models
5. I used the below models for my ensemble-
a. LightGBM - biggest contributor to the ensemble
b. Catboost 
c. Random forest 
d. Gradient Boosting Regression 

**Ensemble-**
1. LAD regression worked for me in comparison to other options like ridge/ optuna based tuning. I used the method suggested in the below link- 
https://www.kaggle.com/code/adaubas/ps-s3e14-stacking-leastabsolutedeviation-reg
2. I blended the top public notebooks with my base models using LAD and prepared my submission
3. I manually adjusted the ensemble weights to a small extent based on my public LB score. 

**Post-processing-**
Thanks to @mattop for the rounding idea, it helped improve my CV score and my leaderboard position as well. I used the idea from his discussion post as below-
https://www.kaggle.com/competitions/playground-series-s3e14/discussion/407327
I post-processed my predictions on the test set after the ensemble (single round of post-processing). 

**Models that did not work-**
1. TabNet regressor
2. Neural networks - I should have focused on my features more to make this work.
3. Linear models
4. XGBoost, especially XGBoost with objective = absolute error (this performed extremely poorly)
5. GAM
6. Extra trees regression

**What I could have done better-**
1. Better quasi-duplicate handling
2. Better feature engineering- I could have tried better secondary features
3. Selected a better final submission. My best submission overall performed slightly worse on the public leaderboard but could have given me a better private leaderboard position. 
4. Better ensemble strategy- I fine-tuned the ensemble weights after the model results based on the public leaderboard score. Perhaps had I not done this, I could have performed slightly better on the private leaderboard

**My key learnings and takeaways-**
1. Rounding off predictions to the nearest training data values - this could be powerful method in many practical assignments in and outside Kaggle
2. LADRegression- I used it for the first time and shall use it well going ahead
3. FLAML - this is quite useful for tabular data. I am predisposed to using PyCaret and LAMA for auto ML based model creation. FLAML is incredibly good for tabular assignments as well.
4. **Even if the CV correlates to the public leaderboard, rely on the CV**. Basically, one should rely on his/ her CV always. 

Finally, wishing all of you the best and see you in the next episode!
Happy learning and warm regards!
