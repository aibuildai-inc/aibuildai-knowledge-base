# Add TABM for Diversity

Competition: playground-series-s5e12
Rank: #4
Source: https://www.kaggle.com/c/playground-series-s5e12/writeups/add-tabm-for-diversity

## 4th Place Solution – Diabetes Prediction Challenge

This competition was a great learning experience, and I owe a huge debt to the discussion page. Honestly, ~80% of my experimentation came directly from ideas shared there. Trying, validating, and sometimes discarding those ideas was the core of my workflow.

#### Key Observations

A common pattern I noticed was that many participants tried TabM, compared it against GBDT models, saw GBDTs outperform TabM on single-model CV, and then abandoned TabM altogether. However, in my experiments, TabM added meaningful diversity to the ensemble, even when it wasn’t the strongest standalone model. This turned out to be crucial.

####  Final Ensemble Overview

My final submission was a weighted ensemble of the following models:

###### Public Notebook Model

Used as a strong baseline and reference point. Added this to the ensemble in the final days. Did not have the oof for this, so had to blend with the public notebook based on few trials and intuition.

###### TabM – Standard Training

Trained normally on the dataset. While not the best on its own, it contributed strong complementary signals.

###### TabM – Oversampled Training (Sample-Weight Approximation)

Since TabM does not natively support sample weights, I approximated them via oversampling.
This significantly altered the learned distribution and improved ensemble diversity.

###### XGBoost with Sample Weights

For XGBoost, I relied on simple target-aware feature engineering:
- For every numerical and categorical feature, I added target mean and count encodings ({feature}_org_mean, {feature}_org_count), with missing values filled using global target statistics.
- Applied frequency encoding to all categorical features.
- Cast categorical columns to category dtype.
- The target was transformed using log1p(y) to reduce skew and stabilize training.

Unlike TabM, XGBoost supports sample weights, which I used to give higher importance to samples after the cutoff ID.
This model carries more weights than the TabM models.

All models were trained on GPU to speed up iteration and allow more aggressive experimentation.
Combination of XGB(4) and TabM(2) models was enough to achieve top 30 in the leaderboard.

#### Models Considered but Not Used

###### CatBoost
I trained a CatBoost model and evaluated it carefully. Using KS statistics and cumulative distribution plots, I found its predictions to be very similar to XGBoost. Due to this redundancy, I excluded it from the final ensemble. That said, with slightly different tuning or weighting, it might have pushed the score even further. In the end, did not have enough time to experiment further on this.

#### Takeaways

- Discussion pages are gold, many ideas may not work directly, but adapting them is invaluable.
- Diversity > single-model strength when building ensembles.
- Models that look “worse” in isolation (like TabM, CATboost here) can still be extremely valuable ensemble members.
- Proper diagnostics (KS test, CDF plots) help avoid redundant models in an ensemble.

Big thanks to everyone who shared notebooks and ideas on the discussion board. Congratulations to the winners!
