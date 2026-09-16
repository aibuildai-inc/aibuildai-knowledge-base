# 6th Place Solution Summary

Competition: PLAsTiCC-2018
Rank: #6
Source: https://www.kaggle.com/c/PLAsTiCC-2018/discussion/75061

6th place solution is a neural network consisting of 3 components:

1. Meta Encoder  - taking as input meta features hostgal photoz, hostgal photoz err, is galactic and summary stats for flux and flux_err as min/max/mean etc.

2. Light Curve Encoder - bidirectional GRU taking as input grouped by day flux, flux_err, detected and time difference

3.  Spectroscopic Redshift Predictor – two fully-connected layers for predicting hostgal specz

Outputs of these 3 components are fed into two last fully-connected layers for predicting class probabilities.

Light curve encoder was pre-trained as autoencoder on test data.  Spectroscopic redshift predictor was also pre-trained on test data objects for which hostgal specz is available.

Augmentations. Three types of augmentations are performed:

1.  flux as a Gaussian with standard deviation flux_err
2.  hostgal photoz as a Gaussian with standard deviation hostgal photoz err
3.  randomly dropping observations

Top submission is an average of 5 cross-validation runs of 3 neural network variations.
