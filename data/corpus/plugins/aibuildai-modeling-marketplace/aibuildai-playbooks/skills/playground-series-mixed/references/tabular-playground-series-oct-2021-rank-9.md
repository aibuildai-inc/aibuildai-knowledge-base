# ✔️ 9th place solution

Competition: tabular-playground-series-oct-2021
Rank: #9
Source: https://www.kaggle.com/c/tabular-playground-series-oct-2021/discussion/284492

Hello Kagglers,

I was very pleased to see I finished in the top 10. I couldn’t have done this without this awesome community and the many great works you shared this month, so thank you! I would also like to share my solution and some of the things I learned. Here's my stack.



######   Preprocessing
I used different scaling methods with my models depending on which one offered the best CV score. I tried out all and had very high hopes for binarization but ended up using Standard/Robust scalers in most cases. Many of my models were trained on just a subset of all features, especially those that took longer to train, like KNeighbors, Multilayer Perceptron, AdaBoost, GBMs, etc. and some were trained on just the binary features alone - e.g. Quadratic Discriminant Analysis. 
I aptly named the features I borrowed from other people's work max_features, luca_features and mottchan_features after their creators 🙂
Here are their notebooks:
- [Tabular play Oct 2021 - Feature selection (idea)](https://www.kaggle.com/maxdiazbattan/tabular-play-oct-2021-feature-selection-idea) by @maxdiazbattan
- [Feature Selection using Boruta-SHAP](https://www.kaggle.com/lucamassaron/feature-selection-using-boruta-shap) by @lucamassaron 
- [TPS - Oct 2021_KMeans++](https://www.kaggle.com/motchan/tps-oct-2021-kmeans) by @motchan 

######   Base models
I used 15 LightGBM models, each trained on 20 seeds and averaged. I made my notebook and dataset public, you can check them out [here](https://www.kaggle.com/adamwurdits/15-lgbms-trained-on-20-seeds-dataset-included). I diversified my stack by adding 32 variants of 15 other types of models. The majority of these were non-tree-based models inspired by the [great notebook](https://www.kaggle.com/davidcoxon/20-model-comparison-oct-tabular-playground) of @davidcoxon.

######   Meta-models
For my level 1 models, I used 9 in total: 3 LightGBMs, Logistic Regression, ElasticNet, Linear Discriminant Analysis, RidgeCV, CatBoost and XGBoost. For my level 2 model, I picked Linear Discriminant Analysis.

I enjoyed this month's competition immensely and hope to see many of you in the next!
