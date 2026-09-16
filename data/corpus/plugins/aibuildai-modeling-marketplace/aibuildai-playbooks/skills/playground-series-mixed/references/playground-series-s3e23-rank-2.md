# #2 Solution | 8 Models Ensemble

Competition: playground-series-s3e23
Rank: #2
Source: https://www.kaggle.com/c/playground-series-s3e23/discussion/450315

First of all, I would like to start with a big thank you to Kaggle for running this episode of the playground series. In this post, I will briefly explain my approach, which most of it can be found in my [notebook](https://www.kaggle.com/code/oscarm524/ps-s3-ep23-eda-modeling-submission).

# Pre-processing

Initially, I modeled the data without any transformation, which produce a descent CV and LB score (about 0.793 CV score and 0.790 LB score). Then, I `log-transform` all of the input features as suggested by @ambrosm in this [post](https://www.kaggle.com/competitions/playground-series-s3e23/discussion/445015). The surprising factor here was that there was a small improvement in model performance in most of the tree-based and boosted-tree models that I considered after the inputs were `log-transformed`.  

# Models & Ensemble

In my [notebook](https://www.kaggle.com/code/oscarm524/ps-s3-ep23-eda-modeling-submission), I trained the following models:

- Random Forest
- Extra Trees 
- HistGradientBoosting
- LightGBM
- XGBoost
- CatBoost

I ensemble those six models with [hill climbing ensemble](https://www.kaggle.com/competitions/playground-series-s3e23/discussion/444784), which gave me a 0.7907 LB score as shown below.



Then, in order to increase the diversity of the ensemble, I ensemble the hill climbing ensemble (of the six tree-based models) with the `Nyström LogisticRegression` model presented in this [notebook](https://www.kaggle.com/code/ambrosm/pss3e23-eda-which-makes-sense#A-few-models). This boost my LB score from 0.7907 to 0.79099 as shown below.



Finally, I decided to include neural network model to the ensemble, which was inspired this [notebook](https://www.kaggle.com/code/sauravpandey11/simple-ann-based-solution) from @sauravpandey11. This boost my LB score from 0.79099 to 0.79101 as shown below.



Unfortunately, I did not select the above submission. I decided to select another another ensemble that had a slightly higher LB score.



# What did not work

I tried a few different things:

- Initially I tried `PCA` as a way to reduce the number of input features to help reduce the model building time. In my notebook, I highlighted that 10 components explain more than 99% of the variability of the data. However, using `PCA` instead of the `log-tranform` features did not help with model performance.

- I also tried `t-SNE` to see if there was a way to separate the two classes but it did not help.

- I also tried clustering as presented in my notebook. But it failed. I even tried target encoding with the clusters but there no improvement in model performance.

# Conclusion

In this comp, ensemble and model diversity were the key to victory!
