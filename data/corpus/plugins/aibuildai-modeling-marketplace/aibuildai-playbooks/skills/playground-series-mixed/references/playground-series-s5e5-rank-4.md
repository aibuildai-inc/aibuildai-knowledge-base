# 4th Place Solution - Ridge Ensemble of 12 Models

Competition: playground-series-s5e5
Rank: #4
Source: https://www.kaggle.com/c/playground-series-s5e5/discussion/582594

First of all, congratulations to everybody for their efforts! A 4th place in my 4th playground competition is really a very pleasant surprise for me. Below i will mention some things from my approach.

A tricky thing about this competition was the CV - LB relationship. A better CV would often result in a worse LB and this was clear from the start. At times, i felt that the correlation was more negative than it was positive and throughout the competition I was never able to achieve a good LB score. At some point i decided to completely disregard the LB and focus on only optimizing the CV score. This turned out to be a good decision!


My final two submissions with the highest CV score had the best private LB scores too. They were:


* Ridge ensemble with 12 models: CV = 0.05868, public LB = 0.05698, private LB = 0.05846, 4th place

* Ridge ensemble with 11 models: CV = 0.05870, public LB = 0.05688, private LB = 0.05847, would have finished in the 7th-10th range.


The ensemble which finished 4th included 3 'level 2' models (extra trees, neural network and LGBM trained on the rest of my OOFs). When i did it i was not sure if it's a good practice to mix level 1 and level 2 models but in the end it gave me an extra 5th decimal for the 4th place. To select the final models for my Ridge ensembles out of 30+ OOF predictions, i used a sequential feature selector with cross validation. 

Below i will mention some individual models which I worked on, most of which were part of the final ensembles. All of them were trained with the same 5 folds, with the exception of Autogluon which i think uses 8 folds and a single XGBoost which i trained with 15 folds.


* **Autogluon**

Autogluon trained for 15 hours and no feature engineering had the best single CV = 0.058800, but at the same time a disappointing public LB score 0.05712. In both my final ensembles this model had a weight > 0.5. I consider it a key part of my solution.


- **GBDT**

Feature engineering did not work at all for me for GBDT models and so I used only the initial features.
The best CV scores achieved for each type of model were:
    - Catboost:  CV = 0.05916
    - XGBoost:  CV = 0.05937
    - LGBM with 'goss' option:  CV = 0.05965 

I trained variations of the above, in some of which i added the original data and in some of them i predicted a transformed target like Calories divided by Duration. I also trained 2 XGBoost models with nested cross validation, using linear regression in the inner folds.
One of them used the predictions of linear regression as a feature and the other predicted the residuals of linear regression.
In another XGBoost i used the per sample weight option to give a big weight to some points which appeared as outliers in the Duration vs Calories scatter plot.


* **Linear Regression**

Linear regression was strong! My best model had CV = 0.05976 and used ~400 features. It's an improved version of [this](https://www.kaggle.com/code/angelosmar1/s5e5-linear-regression-cv-0-05992) notebook i posted during the competition.
This was my favourite model in this competition. Also thanks to @dantetheabstract for a nice linear regression starter [notebook](https://www.kaggle.com/code/dantetheabstract/ps-s5e5-linear-model-starter).


* **Neural Networks**

The architecture for all my NNs was based on [this](https://www.kaggle.com/code/masayakawamata/s5e5-resmlp-cv0-05990) notebook of @masayakawamata. Thanks for this notebook!
I have posted my best NN (CV = 0.05954) in [this](https://www.kaggle.com/code/angelosmar1/s5e5-torch-nn-cv-0-05954) notebook and a few more details about the training in [this](https://www.kaggle.com/competitions/playground-series-s5e5/discussion/582472#3214666) comment. Also thanks to @cdeotte for a nice starter NN [notebook](https://www.kaggle.com/code/cdeotte/nn-mlp-starter-cv-0-0608).

My ensembles included also a NN in which i treated the problem as multi-label classification over 277 labels. The final prediction of this model was the average value, weighted with the predicted probability distribution.

Some other NN's I tried treated all variables as categorical and used one hot encoding or embeddings but generally did not work as good as the other ones. Even though the initial features had a very small number of unique values, treating them as categorical did not work very well.
