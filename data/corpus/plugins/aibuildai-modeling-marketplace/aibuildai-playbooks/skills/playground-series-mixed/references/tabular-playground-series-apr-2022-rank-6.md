# 6th Place Solution

Competition: tabular-playground-series-apr-2022
Rank: #6
Source: https://www.kaggle.com/c/tabular-playground-series-apr-2022/discussion/322622

I was really excited this whole competition since it was my first time finishing in the top 10.

My solution was similar to most of the ones I have seen - an ensemble of RNN and gradient boost solutions.

The main difference between mine was the architecture of the RNN model. I first projected each sequence into an additional 16 dimensions with a linear dense layer - so my data was (n, 60, 13, 16) dimensions. Then I applied a separate GRU network with 4 GRU layers to each of the 13 sequences separately. This stopped the GRU model from overfitting to noisy covariates between each sequence, and allowed the network to converge better. This model performed 0.9839 on the private LB and 0.985 on the public LB.

Ensembling this model with XGBoost, 1D-convolutional, and public LSTM models achieved my final private LB score of 0.98797.

Thanks to everyone for all the interesting discussion and ideas on this competition!
