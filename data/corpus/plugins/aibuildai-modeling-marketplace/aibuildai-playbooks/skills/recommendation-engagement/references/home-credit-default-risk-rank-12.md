# #12 solution

Competition: home-credit-default-risk
Rank: #12
Source: https://www.kaggle.com/c/home-credit-default-risk/discussion/64504

Thank Kaggle and HomeCredit for hosting such an amazing competition. And also thanks for my awesome teammates [@cxlcc][1] [@ouyangxuan][2] [@Yifan Xie][3] [@YL][4] [@Zhiqiang Zhong][5]. We are the second time to team up and have a wonderful time. Our best result's local cv is  0.8047, it is an average of our blending results and stacking results.

# **Feature Engineer** #
In this part, I mainly describe features of [@ouyangxuan][2]’s and my part, and [@Yifan Xie][3]’s group has many features from other different points. Our best single model uses a combination of features from every teammate, and the best local cv of this single model is 0.8021.

## 1. Statistical Features ##
In this part, we mainly use groupby SK_ID_CURR to generate features. 

1)	First we do feature engineer for every single table: we use +-*/ or groupby to generate simple features, such as interest ratio, time interval, days repay late, and etc. To evaluate users' recent behavior, we also groupby the last-k loans of the user to extract features.

2)	Then we merge all the features from every table into application_train table, numerical features (count, max, min, mean, median, std, skew, last, trend, weightd mean), category features (nunique, one-hot count, last). We also try groupby [‘SK_ID_CURR’ + category] features from every table. 

Start with this part of feature engineer, our local cv is about 0.798 – 0.799.

## 2. Model Features ##
After above feature engineer, we think we might have missed some patterns in history loan sequence, so we decide to use models to extract features. We use NNs and LGBMs to generate features from every history table. 

1)	NN features: we use RNNs and CNNs to predict the next loan default prob for every user from his loan history. We believe these NN models can find some time-series pattern from the loan sequence or application sequence.

2)	LGBM features: we use LGBMs to predict whether a history loan is from a default user or not in every table. We treat every history loan from the default user as target 1, then we train LGBM to get the prob, groupby these probs by SK_ID_CURR and merge these into the train table. We think these LGBM models can find some tree split pattern.

After adding these model features, our local cv improves to 0.800-0.801. 

# **Model Ensemble** #
## 1. Blending ##
In the blending part, we mainly concern the diversity from different train input data. We combine all features from every teammate and obtain about 2k+ features. Then we use random 0.5 fraction of these features to generate models, with random kf seeds and random model parameters. We use this strategy to generate about dozens of LGBM models, and the best single subsample model has the cv of about 0.8014. We average the predictions from these models and cv boosting to 0.8037. Then we blend these results with our best single model and the final cv is 0.8040.
## 2. Stacking##
In the stacking part, we mainly concern the diversity from different models. This part is mainly operated by [@YiFan Xie][3] and we have dozens of different models such as lgb, xgb, rf, linear models, and etc. These stacking models give us a cv about 0.8034. In the last day, we fit the blending model oofs from above into the stacking net, and then the stacking cv boosting to 0.8046.

Our best result is an average of stacking and blending results, it has a cv of about 0.8047, PB 0.803.


  [1]: https://www.kaggle.com/cxl923cc "@cxcc"
  [2]: https://www.kaggle.com/oyxuan "@ouyangxuan"
  [3]: https://www.kaggle.com/yifanxie "@Yifan Xie"
  [4]: https://www.kaggle.com/yl1202 "@YL"
  [5]: https://www.kaggle.com/zhiqiangzhong "@Zhiqiang Zhong"
