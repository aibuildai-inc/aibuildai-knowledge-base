# 11th Place Solution

Competition: ieee-fraud-detection
Rank: #11
Source: https://www.kaggle.com/c/ieee-fraud-detection/discussion/111235

Congrats to all the top teams, especially people at the gold zone. This has been a tough competition to us (at least to me 😎).
Our solution was based on stacking. I personally started this competition with stacking about 20 models and after adding more and more oofs (started with Roman's KFold(k=5, shuffle=False) validation scheme with LB~0.9419), 
then I continued by adding oofs from:

*  most of the public kernel models,
*  LGBM based on data created using categorical features + PCA on numerical features,
*  XGB trained trained on different encodings (target, catboost, weight of evidence, ... encodings) of categorical features + numerical features,
*  Lasso trained on different subsets of numerical features (mean and median imputations),
*  Ridge trained on different subsets of numerical features (mean and median imputations),
*  LGBM/Ridge trained on different subsets of count features (each feature count encoded),
*  LGBM/Ridge trained on count features (each feature count encoded) + onehot encoding of counts,
* LGBM/Ridge trained on count features (each feature count encoded) + onehot encoding of counts (dimension reduced using truncated SVD),
* Gaussian Naive Bayes trained on different subsets of count features (each feature count encoded),
* Multinomial Naive Bayes trained on a subset of count features,
* Univariate Logistic Regression trained on a subset of Vs picked using Spearman correlation between them and the target,
* LGBM/XGB trained on categorical features + some numerical features  features (picked using Kolmogorov-Smirnov test),
* a couple of NFFM, FFM models trained on raw categorical features + numerical features using [this](https://github.com/guoday/ctrNet-tool),
* LibFM trained on a subset of categorical features using [this code](https://github.com/jfpuget/LibFM_in_Keras),
* CatBoost trained on different subsets of categorical + numerical features (all features treated as categorical and fed to CatBoost encoding API),
* NN trained on raw data (embedding of categorical data + different imputation methods for the numerical data) with different architectures,  
* LGBM/XGB trained on a subset of 2d, 3d, 4d interactions of categorical features,
* Extra Tree classifier trained on different subsets of PCAed data set (numerical data),
* and KNN trained on a small subset of numerical  features.  

After this, I ended up with about 900+ oofs. By selecting about 120 oofs out of these oofs using RFE (Recursive Feature Elimination) with LGBM as the classifier, I was able to reach 0.9552 on LB.  After that I teamed up with my current teammates where we pooled our features and oofs. ynktk had an XGB trained on about 2000 features that scored 0.9528 and he had created lots of oofs with the same validation method as mine as he worked on his single model. ynktk had been using KFold + brute force feature engineering (various aggregations of the data) from the start and my other teammates were mostly doing FE based on validation using GroupKFold on months so in terms of diversity, it was very helpful. Luckily after a day or two we reached 0.9579. 

I started to search for the m*gic (😎) after reading Konstantin's post about `making them do friendship` and he using CatBoost (to me his magic was most probably about some form of an interaction). First I focused on the interactions of the categorical features especially card1-card6 and addresses then I added Cs and Ds to my interactions. After a lot of experiments with CatBoost I found the feature `card1 + date.day - D1` (treated as categorical) and shared it with my teammates (later Youngsoo Lee found out that `card1 + card6 + addr1 + addr2 + date.day - D1` was a better option). After focusing on this feature and adding new oofs based on it, we reached AUC=0.9649.

Our best stacked model was a 5 time bagged XGB (it was a bit better than LGBM) trained with about 190 oofs (selected from a pool of about 1100+ oofs) that scores 0.9650 on LB with CV=0.9644.
