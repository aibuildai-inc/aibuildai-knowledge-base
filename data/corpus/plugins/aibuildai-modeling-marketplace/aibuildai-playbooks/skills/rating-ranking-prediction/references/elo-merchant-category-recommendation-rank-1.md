# My simple trick for this competition

Competition: elo-merchant-category-recommendation
Rank: #1
Source: https://www.kaggle.com/c/elo-merchant-category-recommendation/discussion/82036

train['final'] = train['bin_predict']*(-33.21928)+(1-train['bin_predict'])*train['no_outlier']

0.015 boost in local cv compare with same feature train directly

I think this trick works because binary is better than regression even though metric is rmse when label is 1&amp;0

like Kaza's post 
https://www.kaggle.com/c/avito-demand-prediction/discussion/59885 
this competition is rmse too,but Kaza use xentropy as their objective

add：
Classifier AUC: 0.914 
no outlier Regression CV:1.545

the most important features for binary_predict from this post
https://www.kaggle.com/c/home-credit-default-risk/discussion/64503#378162
