# #1: A Zoo of Models

Competition: playground-series-s3e11
Rank: #1
Source: https://www.kaggle.com/c/playground-series-s3e11/discussion/399401

My solution is based on the following observations:
1. Only a subset of the features is useful, although it's not completely clear which features belong to the subset. Being in doubt which subset is the right one, we can make models for different feature subsets and blend them.
1. After feature selection, there are lots of duplicates in the training data. We can reduce training time by grouping these duplicates. Training with 3000 groups rather than 360'000 samples speeds up the development cycle massively.
1. If we add the original data to the training dataset, the score improves.
1. `store_sqft`, which has only twenty unique values, looks like a numerical feature, but it is categorical.
1. As most regression algorithms in the library have been developed to optimize rmse, we should make all models predict the transformed target `log1p(cost)` and submit `expm1(pred)`.
1. A "zoo of models", i.e. a diverse ensemble, averages out the prediction errors of the models.
1. Training and test data adhere to the same distribution so that we may trust the cv scores completely.

Point 4 is illustrated by the following diagram. The partial dependency plot for `store_sqft` doesn't look like a typical regression curve - it looks like a categorical feature which has been mistaken for a continuous one. This observation suggests that we might one-hot encode or target-encode the feature.

[dependence]

The diversity of the model zoo can be analyzed with a dendrogram. The dendrogram represents a hierachical clustering of the models where similar models end up in the same cluster. We can easily identify four clusters:
1. four models which were trained with the additional `unit_sales` or `store_sales` feature
1. all random forests and extra-trees models
1. the neural network
1. all gradient-boosting models

[dendrogram]

The bar chart shows the cv scores and training times of all models and ensembles. The final submission is a weighted blend of all 18 models with the weights determined by ridge regression.

[bar chart]

More details can be found in the [source code](https://www.kaggle.com/code/ambrosm/pss3e11-zoo-of-models).
