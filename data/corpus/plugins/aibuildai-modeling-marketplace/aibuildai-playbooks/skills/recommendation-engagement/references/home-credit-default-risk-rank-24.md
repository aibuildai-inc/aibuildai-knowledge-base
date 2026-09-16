# 24th place - Simple Solution with 7 Models.

Competition: home-credit-default-risk
Rank: #24
Source: https://www.kaggle.com/c/home-credit-default-risk/discussion/64548

First of all I want to thank my wonderful teammate Abdel for all his fabulous work, many thanks to all contributors of discussion during the entire competition (Oliver, Tilii, Bojan...), and last but not least, thanks to every grandmasters (Giba, Kaz, Silogram ...) who push us to get beyond our limits. 

This is a brief summary, which will be completed soon for the FE part  (when Google AI competitions are over) !

Our solution is based on diversity of features enginereed datasets (Aggregated, Binarized, Stats, Lags ...). We have construct seven differents dataset,  with a lot of differents features and one unique model  for each one. 

The main reason of this approach is that we first try a big features enginereed dataset, and made diversity with the models but CV was not really increasing. That's why we tried to tackle the problem with this way.

The seven models were really basic :

 -  6 Boosting models  with CV in range [0.792-0.80] and LB in [0.790-0.802]. Each one was optimized with Bayesian search for hyperparameters. And we brute force one with stump trees on a small dataset.
-  1 NeuralNet with CV 0.793 and LB 0.797. (PreLu/BatchNorm/High DropOut) (giving a +0.02 boost)

We stack them with a LogReg CV  0.804  LB 0.806 and a LinearModel CV 0.805 LB 0.807.

So, as you can see, nothing really fancy, just do some features engineering and ***trust your CV*** !
