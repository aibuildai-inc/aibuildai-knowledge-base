# 8th place solution

Competition: reducing-commercial-aviation-fatalities
Rank: #8
Source: https://www.kaggle.com/c/reducing-commercial-aviation-fatalities/discussion/84527

Hi guys,

Let's here talk about my solution for Reducing Comercial Aviation Fatalities.

Features

I already posted here, a kernel that show with only 5 features with could explain more than 90% of variance in data.

[EDA and Simple Unsupervised Analysis](https://www.kaggle.com/brunoguilhermeg/eda-and-simple-unsupervised-analysis-1)

In this way, in my final solution i add just a concatenation(sum,subtraction and square) of this features, something like this:

newFeature1 = Feature1 + Feature2
newFeature2 = Feature1**2
…
After that, i did a normalization on data using min max approach(https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html)

** *Model **

My final solution, count on 5 models(KNN,Logistic Regression,XGB,LightGBM,NN). I build just a stacked net to try learn the weights composition of each model in final predict. In this way, i did a cross validation with 5 folds plus over sampling in train data to build each model.


That's it guys, i hope you enjoy.
Bruno Guilherme
