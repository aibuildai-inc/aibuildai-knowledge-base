# 19th Place - Ensemble of 29 models

Competition: playground-series-s6e4
Rank: #19
Source: https://www.kaggle.com/c/playground-series-s6e4/writeups/19nd-place-ensemble-of-29-models

**Dear Friends!**

Congratulations to all the winners and all who improved their own results in this Competition.
I was glad to take part in the Competition. My laptop became my constant companion everywhere last month).

**Solution**

My solution is an ensemble of 29 models. Meta-model – RidgeClassification with hyperparameters tuned by Optuna. The features used to train the meta-model include not only class probabilities, but also the class predicted by the models and the Chris' exact formula.

**Models and Feature Engineering**

I used the following models: XGBClassifier, LightGBMClassifier, CatBoostClassifier, HistGradientBoostingClassifier, RealMLP_TD_Classifier.  
Hyperparameters were tuned for some of them with Optuna, but not for all. All models were trained on 5-folds CV.

The models were trained on different sets of features:
    - The initial set of features + Ohe
    - The initial set - of features + Ohe
    - The initial set of features + Ohe + Chris formula
    - The initial set of features +  magical feature (combination of all important features)+Ohe
    - The initial set of features + Frequency Encoding
    - Combinations of categorial features+ohe
    - Digits ) +initial set of features
    - Binning numerical features +  nested target Encoding 
    - Nested target Ecoding (categorical features) 
    - TargetEncoding using Original dataset
    - 2 models  were taken from open from public notebooks: https://www.kaggle.com/code/arshikhan810/xgb-with-te-t4 and https://www.kaggle.com/code/rawashishsin/s6e4-highest-score-xgboost-cv-0-98109

**What didn't work**
I tried to do bias tuning. This gave an increase on cross validation, but lowered the rating on private leaderboard.

**My notebook:**
https://www.kaggle.com/code/elenkapetrova/ps-s6-e4-metamodel-ridge


Many thanks to @cdeotte, @yekenot, @rawashishsin, @arshikhan810!
