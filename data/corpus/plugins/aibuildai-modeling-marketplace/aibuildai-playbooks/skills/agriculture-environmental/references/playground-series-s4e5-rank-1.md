# #1st place solution

Competition: playground-series-s4e5
Rank: #1
Source: https://www.kaggle.com/c/playground-series-s4e5/discussion/509043

Thank's to Kaggle for organizing tabular competitions like this one. I learned a lot from those competitions. I learned a lot also from public notebooks and discussions, thank you everyone for sharing so much.

Because this dataset contains a sum of [Poisson](https://en.wikipedia.org/wiki/Sim%C3%A9on_Denis_Poisson) distributions, the winner of this competition had to be from France. Thank you M. Simeon Denis Poisson for giving us such a tool, and thank you for giving me a extra motivation during this competition.

Thank you to all competitors, you are really challenging, playing with you is a pleasure. Congrat's to @mdoroch (I will contact you, you were very impressive), @lashfire, @mattop , @tilii7 : your submissions during last days  stressed me so much !

My solution is  :
* an ensemble ajusted with a ridge regression `positive = False, fit_intercept = False`.
* and ajusted with 3 repeated oof predictions of a variety of models produced with catboost, xgboost, lightgbm and 2 public notebooks
    * **Auto Gluon Starter** by @mfmfmf3 available [https://www.kaggle.com/code/mfmfmf3/autogluon-starter](https://www.kaggle.com/code/mfmfmf3/autogluon-starter)
    * **Flood Prediction LGBM** by @igorvolianiuk https://www.kaggle.com/code/igorvolianiuk/flood-prediction-lgbm
    

**EDA** helped me to understand that original dataset was completly different from train dataset. EDA helped me to understand that the sum of 16 of 17 orginal features was still a Poisson distribution in train dataset, but sum of 18, 19, and 20 original features was not (neither in test dataset). And EDA helped me to find interesting features, correlated to target (thank you @ambrosm for the most important feature)


I took care for my own GBMs to **feature selection** :

* usefull features : sum of each lign, std of each lign, max of each lign, sorted original features (thank you @siukeitin for [Sorting along the feature axis as "feature engineering"](https://www.kaggle.com/competitions/playground-series-s4e5/discussion/499579)'s discussion, number of variables which have a value higher than 6, or than 7, or than 8 (count features) : 

* useless features : original features, skewness of each lign, kurtosis of each lign, redondant features. I dropped them.

* magic feature : because target could be seen as a discrete variable, I used a **target encoder**, and `train.groupby("sum")["FloodProbability"].std()` was an interesting feature (where `train["sum"] = train[test.columns].sum(axis=1)`)

* I used **permutation feature importance** technique explained in [scikit-learn](https://scikit-learn.org/stable/modules/permutation_importance.html) or in [Kaggle](https://www.kaggle.com/code/dansbecker/permutation-importance) to detect useless features. I used **backward** technique too.

**Variety of models**
I trained more than 30 GBM's, with various sets of features : sometimes with sorted original features, or sometimes with count features `"nb_inf4", "nb_inf3", "nb_inf2", "nb_sup6", "nb_sup7", "nb_sup8"`, sometimes with target encoder, sometimes without.
I used also a **target transformation** sometimes : there was a strong signal (sum), and a noisy signal (deviance from sum). My transformed target was : 
`train["target_transf"] = train["FloodProbability"] - df[original_features].mean(axis=1) * 0.1`
or `train["target_transf"] = (train["FloodProbability"]*400 - train[original_features].mean(axis=1) * 40).astype(np.int16)`
Thank's to @act18l for https://www.kaggle.com/competitions/playground-series-s4e5/discussion/499263#2787165

During the first 10 days, I used only default parameters (with early stopming) and didn't optimize any GBM, concentrated on my features and my ensemble loop.
After, hyperparameters were optimized with optuna by using only my kaggle quota of GPU. I optimized only the following hyper parameters (with max timeout = 5400 seconds, it's a lot) : 
* depth for catboost and xgboost, 
* num_leaves for lightgbm, 
* alpha, lambda, min_child_weight or min_child_samples for xgboost and ligntgbm 
* bagging_temperature, random_strength for catboost
* subsample, colsample_by_node for xgboost and lightgbm

I used default **grow_policy** of each GBM method (xgboost, catboost and lightgbm have each a different default grow_policy value, it gives variety of predictions), I used default learning_rate or .1, I used early_stopping (od_wait in catboost) to fit n_iterations. 

**Ensemble**
During the first two weeks, I trained my ensemble with **LinearRegression** and positive = True and fit_intercept = False. Until I tried **Ridge** and positive = False (still fit_intercept = False) which gave me a little improvement. To have a robust ensemble, I trained all my models on 3 repeated kfold, and select by CV the features for my ensemble.

At the end of week 2, I made a submssion with a blind blend using [https://www.kaggle.com/code/mfmfmf3/autogluon-starter](https://www.kaggle.com/code/mfmfmf3/autogluon-starter) output. I had a .86939 public score better than .96934 my previous one. Few days later, I had a first complete oofs file from AutoGluon. Then I fitted carefully a new ensemble, and get .86941 score at the end of week 3.

Then I decided to get 3 oofs AutoGluon files and to fit AutoGluon with some other features (3 more oofs). I trained in parallel autogluon [here](https://www.kaggle.com/code/adaubas/pss4e05-copy-autogluon-starter) to get 6 oofs files. And then I fitted my final ensemble which score .86943 on public LB. 

In conclusion, I have success by combining :
* EDA
* feature engineering
* variety of GBM
* AutoGluon
* a strong CV for my Ridge ensemble

I'm proud to have made only 2 submissions between the 18th of may and the end, to win on public and private LB. My CV was highly reliable to public LB since the first day, I trusted my CV and didn't need to submit.

My final ensemble is available here https://www.kaggle.com/code/adaubas/pss4e05-ensemble-with-ridge.

I'm so happy ! Waouh
