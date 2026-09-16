# #13 approach- simple ensemble with tuning

Competition: playground-series-s3e10
Rank: #13
Source: https://www.kaggle.com/c/playground-series-s3e10/discussion/396319

Hello all,

Firstly, many thanks to Kaggle for the assignment and many thanks to all the contributors for making the experience great! Special thanks to @sergiosaharovskiy, @pourchot and @paddykb for their public contributions.

**My approach:-**

**Feature engineering:-** 
a. Used multiplicative and additive features (example:- Mean_Integrated *  SD and Mean_Integrated + SD) as total features
b. Selected features using mutual information, correlation and permutation importance, chose top 25, 30, 32, 35, 40, 45 features for the model (final submission contained 35 features). I also chose some features based on leaderboard probing and based on CV scores manually at times. 
c. Did not do any explicit outlier treatment and feature transforms like log, power transforms, logit, etc. 
d. Did not resort to any scaling too, based on CV
e. Precluded the original data (CV score was slightly lower on model training)

**Models:-**
Used the below classifier algorithms:-
a. LGBM
b. XGB
c. CatBoost
d. Gradient Boosting
e. Histogram Gradient Boosting
f. Random Forest
g. Logistic Regression with robust scaler
h. Generalized additive model (thanks to @paddykb and @pourchot for the public work)

**Model tuning**
I tuned the parameters with optuna and perturbed the tuned parameters to check the impact on the CV score. I had to adjust some parameters based on this exercise (in LGBM and GBM classifiers). **I did not tune the scale_pos_weight** for this assignment. I resorted to tuning just the basic parameters as below-
a. learning rate
b. max_depth/ depth
c. reg_alpha
d. reg_lambda
e. n_estimators

**Model training**
a. CV strategy:- Repeated stratified KFold 10x5
b. Early stopping rounds:- 120

**Ensemble and calibration:-**
a. **I calibrated only the GAM results using isotonic regression**
b. I built an ensemble using optuna (1000 trials)
c. I adjusted the weights for GAM (I overweighted it manually and underweighted random forest and logistic regression for my final submission)

**What did not work:-**
a. TabNet classifier
b. Neural Network- MLP
c. Calibrating all probability predictions across all models
d. Platt scaling for GAM calibration

**What I could have done better:-**
a. Better feature engineering - I used only multiplicative and additive features, I could have used ratio/ log/ other transforms
b. I could have tuned my models in a better manner, and not rely on optuna for most classifiers

**Note on GAM:-**
From my work experience, I may posit that GAM is a great method for classification and regression problems alike. I used GAM for the first time on python in this challenge, but have used it frequently on SAS. I encourage SAS users to use [PROC ADAPTIVEREG](https://documentation.sas.com/doc/en/pgmsascdc/9.4_3.4/statug/statug_odsgraph_sect109.htm), [PROC TRANSREG](https://documentation.sas.com/doc/en/pgmsascdc/9.4_3.4/statug/statug_odsgraph_sect092.htm) and [PROC GAM](https://documentation.sas.com/doc/en/pgmsascdc/9.4_3.4/statug/statug_gam_syntax01.htm) for various assignments. 

Best regards and happy learning!
