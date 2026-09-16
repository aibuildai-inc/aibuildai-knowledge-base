# #5 Solution | five models + LADRegression

Competition: playground-series-s3e16
Rank: #5
Source: https://www.kaggle.com/c/playground-series-s3e16/discussion/416769

Hi all,

First, I would like to start with a big thank you to Kaggle for running this episode of the playground series. In this post, I will briefly explain my approach, which most of it can be found in my [notebook](https://www.kaggle.com/code/oscarm524/ps-s3-ep16-eda-modeling-submission/notebook).

# Preprocessing

I did not preprocess the data at all. 

# Feature Engineering

There were some awesome features engineering post in the competition. However, in my model only a few features improve the performance of my models. The features that help are:

```
X['Meat Yield'] = X['Shucked Weight'] / (X['Weight'] + X['Shell Weight'])
X['Shell Ratio'] = X['Shell Weight'] / X['Weight']
X['Weight_to_Shucked_Weight'] = X['Weight'] / X['Shucked Weight']
X['Viscera Ratio'] = X['Viscera Weight'] / X['Weight']
```

Notice that the above features were suggested in this [post](https://www.kaggle.com/competitions/playground-series-s3e16/discussion/415721) by @pandeyg0811.  

# Modeling & Ensemble

I considered a 10-fold CV framework with the following five models:

* GradientBoosting
* HistGradientBoosting
* LightGBM
* XGBoost
* CatBoost

And I ensemble those five model predictions using [LADRegression](https://scikit-lego.netlify.app/linear-models.html#Least-Absolute-Deviation-Regression). Notice that I rounded the five model predictions to the nearest integer before I ensemble them with `LADRegression` (this boost the CV a lite bit). Then, I re-run the above framework with different seeds and ensemble the predictions by computing the mode of the predictions for each of the `id` in the `test` dataset. 

# What didn't work

* I tried several features (some of them I engineered on my own, and other suggested in a few discussion posts) but only the ones listed on the feature engineering improve model performance.
* I also tried [FLAML](https://microsoft.github.io/FLAML/) but due to my lack of experience with the framework, I could not build a decent enough model. 

# Conclusion
In this competition, my local CV aligned pretty well with the LB and private LB. So, the moral of the story is to trust on your CV.
