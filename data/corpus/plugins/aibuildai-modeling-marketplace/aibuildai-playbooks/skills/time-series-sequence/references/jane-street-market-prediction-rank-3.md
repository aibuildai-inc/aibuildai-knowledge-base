# 3rd place solution: Ensembles of deep (49 layer) MLPs

Competition: jane-street-market-prediction
Rank: #3
Source: https://www.kaggle.com/c/jane-street-market-prediction/discussion/224713

After many initial experiments with more or less „complicated“ approaches I ended up with almost no feature engineering, no CV, but an esemble of 15 deep MLPs, each of them consisting of 49 layers. 

**Data:**
- Remove first 85 days
- Rows with weight=0 were included
- NaN filling: median conditioned on f0 (I think this was actually my best idea throughout this competition :D )

**Model:**
- input block: batchnorm + logarithmic feature extension (→ 130 std features, 130 log features)
- block 0: 3 dense layers with 100 units, batchnorm, dropout (0.35) and mish activation
- block 1-23: 2 dense layers with 100 units, batchnorm, dropout (0.35) and mish activation
- skip connections between Block 0 and Block i (i=[1,..,23]
- output: Dense layer with 5 units (all 5 ‚resps‘) - sigmoid activation
- 15 ensembles of these deep models
- action: mean over all 15x5 outputs; threshold = 0.5

**Training:**
- batch size = 30k
- Adam with lr = 1e-3
- 200 epochs
- loss: mixture of BCE (label smoothing = 0.1) and a „utility-like“ loss

**Inference:**
The models are converted to tf-lite interpreters. This really speeds up submission. Otherwise it never would have been possible to evaluate so many deep models.

**Validation:**
The basic model shape and the hyperparameters where chosen from a simple train/ validation split on my local machine. In between I also used a GKF CV but I wasnt really happy about it, so I went back to this very simple train / validation split (train: 85-300, val: 350-500).
To reduce the impact of different seeds, I rerun my tests up to 8 times. This gave me quite a good estimate of how new implementations really perform.
For the final model, the number of epochs for training were found by retrain the models and submit them. So one could say I used the public LB data for early stopping. As we have 1 million rows in the public set, this seems legit to me.

**Ensembles:**
It is often said that ensembles only improve the performance if the models learn something different, i.e. if they are trained with different data. I think this is true in general, but here the situation is a little different. We saw an extreme dependence of seeds for different public models. I also found this for my single models. Using many models trained from different seeds kinda solves this issue. Like fighting fire with fire, fight randomness with randomness.
