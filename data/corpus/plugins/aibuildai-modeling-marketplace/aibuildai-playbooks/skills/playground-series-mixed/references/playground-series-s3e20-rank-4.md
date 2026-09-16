# 4th place solution: using PCA (least shakeuped solution)

Competition: playground-series-s3e20
Rank: #4
Source: https://www.kaggle.com/c/playground-series-s3e20/discussion/433567

[The solution notebook is available here](https://www.kaggle.com/code/kdmitrie/pgs320-pca)

Thanks to the organizers of this interesting competition, and thanks to all participants for the discussions!
I think there was much valuable information to learn and improve!

# The pipeline
I started this competition by building a pipeline to make CV and test different approaches. This pipeline consequently used each available year (2019, 2020, and 2021) for validation while the others were used for training. Finally, all the data was used for training and prediction of the 2022 emissions.

Please, refer to [the source code](https://www.kaggle.com/code/kdmitrie/pgs320-pca) for details.



# Using PCA
The core of my solution is using PCA. I decided to use 6 components, as I reported in [this discussion](https://www.kaggle.com/competitions/playground-series-s3e20/discussion/429278) by @ambrosm .
Each of the 6 components is processed independently by some (abstract) algorithm, and the results are taken to inverse PCA. Finally, we have the predictions.



# Algorithms
Each algorithm I used has a base estimator as a parameter, which is used to make a prediction. With such an architecture, it's possible to switch between them and optimize CV score.

### ✓ PCA1 Algorithm
This algorithm is very simple: it takes `emission` data only and processes it with an estimator. It was reported in many discussions that all other columns are not needed. So this is an implementation of this idea.

### ✓ PCA2 Algorithm
This algorithm is different: it uses **all columns**. PCA is applied separately to each of them, i.e., we have `n_columns` * `n_pca_components` input features.

### ✓ PCA_SARIMA Algorithm
Instead of simple estimators, I used 6 independent SARIMA models in the PCA1 Algorithm architecture.

**The best algorithm (both local CV and private LB) was the PCA2 Algorithm.**

# Estimators
I used Ridge/Lasso regression, RF, XGB, CatBoost, etc. There was not much difference between tree-based estimators, and they gave better results than linear regression. So I choose XGBRegressor(n_estimators = 300, max_depth = 4, learning_rate = 0.01, subsample = 0.5).

# Postprocessing
I was inspired by @ambrosm 's [post](https://www.kaggle.com/competitions/playground-series-s3e20/discussion/428791) and other discussions, which propose to multiply the result by the constant. So, I made predictions with different multipliers: 1.00, 0.95, 1.05, 1.07, and 1.08.

However, somehow I chose the worst submission of these five as my final prediction!


It would be the 1st place, if I didn't use the multiplier at all!
