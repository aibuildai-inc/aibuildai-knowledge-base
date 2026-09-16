# Rank14 approach - Grand blend of diverse models

Competition: playground-series-s6e1
Rank: #14
Source: https://www.kaggle.com/c/playground-series-s6e1/writeups/rank14-approach-grand-blend-of-diverse-models

Hello all,

Thanks to Kaggle for a wonderful stable dataset this time around! This episode was one of the best Playground dataset I have seen since the [July2024 - Insurance cross selling](https://www.kaggle.com/competitions/playground-series-s4e7) episode. The CV scheme worked perfectly and all improvements locally translated wonderfully to the leaderboard. Many thanks to the forum contributors (except for the blind blending junk) for some good ideas across kernels and discussions. Also, best wishes to the winners too!

## CV scheme details 

I followed a simple 5-fold scheme and saved the folds across all my experiments. <br>

`cv = KFold(n_splits= 5, shuffle = True, random_state = 42)`

All my single models and ensembles followed this CV scheme. I did not do any full-refit type of training and relied on the CV scheme all throughout. 

## Feature engineering

Since day-1, we were clear that this dataset was stable. Hence, adding meaningful features was key to creating a diverse ensemble and a large blend of single models. One may engender a successful ensemble in 3 ways- 
1. Similar model algorithms with varied features 
2. Varied model parameters across chosen algorithms 
3. Custom parameters, loss functions and models 
4. Innovation in ensemble design

Strong feature engineering is the backbone behind all of this and is cynosure to a successful ensemble all throughout. 

My feature engineering for this competition included the below-
1. n-gram features (1-2-3-4 grams were considered)
2. Additive/ multiplicative type of features 
3. Varied datatypes (some / all columns as category dtype)
4. Public features across shared kernels with some personal modifications 
5. Adding multiple copies of train / original datasets as rows / columns - these are normally discussed and shared across public work too

Most of my single models included between 30-200 features based on the CV score and code-runtime and GPU consumption. 

## Single models 

My single model suite consisted of the below. I shall resort to general comments here as we experienced a 100% CV-LB linkage.

|Model algorithm type| Comments|
|------ | ------- |
| TABM regressor | - Best single model by a distance <br> - All variants and model parameter adjustments worked wonderfully | 
| Real-MLP regressor | - Very reliable single model and provided a lot of diversity and quality <br> - All variants and model parameter adjustments worked wonderfully | 
| XgBoost regressor | - Best single model option among boosted tree model category <br> - Adjusting the loss function helped somewhat | 
| LightGBM regressor | - Decent performance but was overshadowed by XgBoost regressor <br> - Adjusting the loss function helped reasonably but performance was not as good as XGB anyway | 
| Catboost regressor | - Did not perform well at all despite a lot of effort with features <br> - I resorted to a set of 4-5 catboost models for diversity but stopped pursuing this model option after a while for other better options  | 

<br>I considered the below loss functions for my models -
1. L2 loss - most common and provided a good enough model across boosted trees 
2. Huber loss - provided a bit of diversity and resulted in a CV slightly better than the MSE option across boosted trees 
3. Weighted MSE loss function - the 13th place solution has described this as well quite well. My code does not include as many levels / categories but is on similar lines. This provided a decent boost to the CV score for XgBoost but not for LightGBM. 
4. I trained my TABM and Real-MLP regressors on standard L2 losses only

## Model ensemble

I resorted to a 3-stage ensemble process including -
1. Autogluon ensemble stack using some / all of the single models as base 
2. A second stage blend with ridge / hill climb that combined all the autogluon models and selected single models 
3. A simple average of the model options in point-2 for my submission

## Key learnings 

1. Trust the CV 
2. One does not need to submit a lot if one experiences a good CV-LB relation. Whatever improves the CV is most likely to improve the LB as well
3. Diversity is key in an ensemble and not necessarily the individual model quality. **Quality adds value, so does diversity**
4. Avoid blind gambles with public work - ingest and consume public work correctly rather than misusing them with soft vote manipulations

Finally, wishing all of you the best for your personal and professional endeavors and your onward Kaggle journey! All the best for the February-Playground episode as well!!

## Past Playground solutions 

I have added a few of my past Playground solutions as reference, this one draws inference from a lot of them too -

1. [Nov-2025](https://www.kaggle.com/competitions/playground-series-s5e11/writeups/rank8-approach-trust-the-cv-score)
2. [Oct-2025](https://www.kaggle.com/competitions/playground-series-s5e10/writeups/4th-place-residual-xgboost-meta-nn-hill-clim)
3. [Aug-2025](https://www.kaggle.com/competitions/playground-series-s5e8/writeups/rank-3-public-rank-5-private-approach)
4. [June-2025](https://www.kaggle.com/competitions/playground-series-s5e6/writeups/ravi-ramakrishnan-rank-28-approach-diversity-and-c)
5. [April-2025](https://www.kaggle.com/competitions/playground-series-s5e4/writeups/ravi-ramakrishnan-rank-4-approach-lots-of-features)
6. [Feb-2025](https://www.kaggle.com/competitions/playground-series-s5e2/writeups/ravi-ramakrishnan-rank-2-approach-a-century-of-com)

Best regards, <br>
Ravi Ramakrishnan
