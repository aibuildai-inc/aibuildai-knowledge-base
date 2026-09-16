# #3 : GAM & GBM

Competition: playground-series-s3e10
Rank: #3
Source: https://www.kaggle.com/c/playground-series-s3e10/discussion/396346

Hi,

3rd solution is a blend of XGBoost, LightGBM and Generalized Additive Models [here](https://www.kaggle.com/code/adaubas/pss3e10-3rd-place-solution-gam-gbm) :

- with from 4 to 10 folds Stratified CV
- with log transformation of some features
- with some interactions features
- with forward selection of interactions features and values permutations of predictions to do backward features selection and to drop useless features and to avoid overfitting.
- with weak learners for XGBoost and LightGBM : only 5 ou 6 leaves
- with overfitting control by computing difference between *val_logloss* and *trn_logloss* for each fold, to fit regularization hyperparameters in XGBoost and LightGBM, **with the hands** (See @ambrosm in https://www.kaggle.com/competitions/playground-series-s3e9/discussion/394592)
- hence without optuna or other optimization tools
- without calibration of predictions, because by CV I saw it was useless
- without original data.

- with a little mistake in final submission in diversity of models, which costs me the second place (see difference betwwen version v40 & v39 - v40 was what i wanted to do but I did v39).

By computing difference between *val_logloss* and *trn_logloss*, we can see that there was less over-fitting with GAM than with GBMs.

I tried RandomForest and ExtraTrees, but my CV was only 0.033.<br>
I tried to fit a LogisticRegression without sucess, I wasn't able to add interactions with polynomialfeatures, thank's @ambrosm for your [solution](https://www.kaggle.com/code/ambrosm/pss3e10-7-winning-model), I'll read it carefully.


**Special thank's to @paddykb for GAM** in this [notebook](https://www.kaggle.com/code/paddykb/ps-s3e10-gam-finger-on-the-pulsarrrrr) and log transformation of some features.
Thank's to @mateuszgrzybpl for pyGAM in python (pygam) in comments [here](https://www.kaggle.com/code/paddykb/ps-s3e10-gam-finger-on-the-pulsarrrrr/comments)
And thank's too to @pourchot for GAM with pyGAM [here](https://www.kaggle.com/code/pourchot/pygam-for-generalized-additive-model)
