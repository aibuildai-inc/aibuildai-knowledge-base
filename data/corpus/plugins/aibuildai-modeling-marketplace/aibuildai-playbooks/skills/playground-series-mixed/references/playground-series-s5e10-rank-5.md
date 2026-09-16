# 5th Place - One Hundred Folds!

Competition: playground-series-s5e10
Rank: #5
Source: https://www.kaggle.com/c/playground-series-s5e10/writeups/5th-place-one-hundred-folds

Thanks everyone for a fun playground competition. In this playground competition, I did not create 100s of diverse models (and stack and hill climb) like I have done in the past. I only created 2 strong models with lots of variations of these 2 models. Namely, I created XGB and TabM.

# XGBoost
In my final submission I included multiple variations of my starter XGB notebook [here][1]. Also I added OOF from multiple TabM as new features thus turning (some of) my XGB into a stage 2 meta (stacking) model.

# TabM
In my final submission I included multiple variation of @masayakawamata TabM [here][2]. Specifically, I converted the TabM to predict residuals over the original dataset generating function (similar to my XGB starter). And I included @yunsuxiaozi TabM [here][3] (I used these TabM as OOF features to my XGB)

# 100 Folds
NN benefit from training with more data and blending multiple copies of itself (with different seeds), so I retrained all my TabM with 100 folds. And I trained XGB with the same 100 folds, so that I could stack XGB over TabM.

# Stacking
After retraining the original 2x TabM with 100 folds and another variation predicted over residuals with 100 folds. I then trained my XGB with these OOF as new features

# Hill Climbing
After training 2xXGB, 3xTabM, and 2xXGB stacked over 3xTabM, I blended these 7 models with Hill Climbing.

# Pseudo Labeling
I boosted my CV and LB by retraining my TabM with pseudo labeled test data (from my 7x model ensemble). (i.e. first I built 7 model ensemble without pseudo label, then I pseudo label test, then i retrain TabM w/ pseudo labeled test added, then I make a new better 7 model ensemble using the pseudo labeled TabM).

# Add Top Public Notebook 50%
Lastly I blended my (7 model) ensemble with 50%/50% of the best public notebook (for 1 final submission while the other final submission was only my 7x model ensemble).

[1]: https://www.kaggle.com/competitions/playground-series-s5e10/discussion/610828
[2]: https://www.kaggle.com/competitions/playground-series-s5e10/discussion/610792
[3]: https://www.kaggle.com/competitions/playground-series-s5e10/discussion/611405
