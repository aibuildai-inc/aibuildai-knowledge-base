# # 7 approach| Hand-picked features and simple ensemble

Competition: playground-series-s3e11
Rank: #7
Source: https://www.kaggle.com/c/playground-series-s3e11/discussion/399463

Hello all,

Firstly, I wish to thank Kaggle for this episode of the ongoing playground challenge season. Also, I wish to thank all the participants who contributed regularly and made this an interesting and rewarding experience. I wish to extend since congratulations to @paddykb and @janmpia for their recent progression to the kernels master tier as well and shall thank them specifically for their valuable contributions throughout the past 2 weeks. 

My approach could be summarized as a **ridge ensemble of tree-based and GAM models using features selectively in each base model**. This is illustrated as below- 

**Feature engineering**-
1. The training set was incredibly noisy and had lots of quasi-duplicates too. I expected lots of churn while doing my initial EDA and hence, decided to approach the problem by using an ensemble of several models with different feature choices. I realized that this approach had a positive impact on my CV and my public leaderboard position too and decided to use this throughout the competition.
2. Thanks to @janmpia, I tried and used the mean features too in some of my base models. Please peruse his work (kernel and discussions) to know more in this regard- https://www.kaggle.com/competitions/playground-series-s3e11/discussion/399393
2. I retained **store square feet, cars at home and florist** in all my models and hand-picked other selectively for all my base models. Each base model choice had 4-6 features, including the above 3 features always and 1-3 features from the remaining ones. 
3. I did not do any further feature transforms/ scaling

**Target-**
1. I used np.log1p(train.cost) and an RMSE metric in my regression analysis. I think this is easier than formulating a custom metric and objective in my models
2. While training the model, I inverted the predictions to facilitate np.expm1(preds) to get back the predictions in the desired format

**Base Models-**
1. I used 20+ base models, comprising of hand-tuned XGB, LGBM, Catboost and GAM regressors in total with hand-tuned parameters, mainly adjusting the learning rate, max_depth, reg_alpha and reg_lambda. I also used early stopping while fitting the model too. 
2. I used R to prepare my GAM. My code is similar to @paddykb. Thanks to him for the work in the current assignment too. 
3. My CV strategy was a simple target encoding, with 10-20 target bins with a pd.qcut(train.cost). Some of my base models were built on 10 bins while others on 20 bins. I used a 10x3 repeated stratified k-fold thence to design my CV using the target bins to build the groups. 
4. I used the **original data in all folds** entirely to build the model training fold, while the evaluation (dev-set) was the corresponding competition fold. Using the original data in this manner improved my CV score and my public leaderboard position quite well

**Ensemble-**
1. My ensemble meta layer was a simple ridge regression (5000 iterations, random_state = 42) to merge the predictions. 
2. I used an optuna ensemble too at the start of the assignment, but ridge regression outperformed optuna and was used for the final submission

**Models that did not work-**
1. TabNet regressor
2. Neural networks - I should have focused on my features more to make this work.
3. Linear base models

**What I could have done better-**
1. I could have used quasi-duplicates to create a sample weight like the winning solution and this would have helped me reduce my training time. I am perplexed how I missed out on this obvious point
2. Better feature engineering- I could have tried better secondary features 
3. I could have used ensemble models like extra-trees and random forest. I did not use them throughout and relied on the GBM based tree models only

**My key learnings and takeaways-**
1. From the winning solution, I take away the importance of using sample weights. I know this and have used it before, I just did not use it this time. I will be careful to adopt this going ahead whenever feasible
2. I should be open to trying a wider model suite rather than relying on a limited choice of ensemble trees for most work
3. R is a powerful language for such tabular assignments, I should improve my R skills and use it more going ahead for such assignments

Finally, wishing all of you the best and see you in the next episode!
Happy learning and warm regards!
