# 9th Place Solution, 9 Models in the Ensemble

Competition: playground-series-s5e5
Rank: #9
Source: https://www.kaggle.com/c/playground-series-s5e5/discussion/582596

It's been at least a year since I've done Kaggle competition. I was about to commit until the end but eventually I had to leave midway, partly because work priority took over, and partly because that Playground Series is the same as ever for good and for bad. Good to know that I'm not rusty at all though :)

# Feature Engineering

There are four feature engineering that I've done:
1. I encoded `Sex` feature with my trusty encoder for all models: M-Estimate Encoder.
2. For neural network and linear regression, I added the inverse of each feature into the model. For example: $$\frac{1}{Sex}\text{, }\frac{1}{Height}\text{, etc.}$$
3. For linear regression specifically, I had to use scikit-learn's `PolynomialFeatures` to generate the polynomial combinations of the features including their interactions. This is because linear regression needs a lot of features to perform relatively well, while it's unnecessary for neural network since you can just set the units in the first layer as high as you want for that purpose.
4. On the other hand, for ridge, I used standard scaling and Nystroem with polynomial kernel for similar reason as above.
5. If you want to generate thousands of features in relatively quick speed without killing your kernel (as long as you don't go overkill on picking the parameter), you can use random forest embedding. Since it's unsupervised, you can train it on concatenation of train and test dataset (and also the original dataset if you want). I've only had the chance to use this on plain CatBoost sadly. The goal here isn't to find the absolute best single model; it's just to provide diversity for the ensemble.

# Models

As mentioned previously, I used neural network (Tensorflow), linear regression, ridge, and plain CatBoost. Other models that I use without any feature engineering except of encoding are Random Forest, XGBoost, LightGBM, and two CatBoost with different bootstrap types. Those 6 with the exception of Random Forest are tuned with Optuna. Overall, that's just 9 models. I certainly could've gone more like what others had done :)

# Ensembling

I used Ridge to check the local CV score first and see what model I need to exclude. After that I used Optuna to find the actual optimal weight. I was about to try Hill Climbing but I got too lazy before I dropped.

# Things to Note

1. Original dataset may not improve the CV by significant amount, but I always find that it's better to include it than not.
2. Sometimes it's not always good to tune all your models to their absolute best. While improving CV of single model is good, some models shouldn't have the best parameter to give the best diversity for the ensemble. I had this experience with Random Forest.
3. Always make sure your CV is robust. If you want to include original dataset, always concatenate it after you split the dataset into train and validation instead of before. You really want to make sure the validation dataset reflects the test dataset.
4. I don't know how it is for others but my neural network performs at its best without residuals. Its layer structure is just [256 > 64 > 16 > 4 > 1 (output)] anyway with batch normalization before activation function in each layer.
