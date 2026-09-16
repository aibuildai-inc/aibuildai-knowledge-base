# 1st place DAE training code

Competition: tabular-playground-series-feb-2021
Rank: #1
Source: https://www.kaggle.com/c/tabular-playground-series-feb-2021/discussion/222745

I am sharing my code to train my DAE model. [Link to the code](https://github.com/ryancheunggit/Denoise-Transformer-AutoEncoder)  

In short, the network uses stacked transformer encoders rather than your typical `linear->relu` hidden layers. I have some more documentation on my thought process in the repo too. Please refer that to for detail. 

With the features extracted from this model, a single ridge regression can give 0.8412 RMSE in a 5 fold cross-validation setting. 

[network]
