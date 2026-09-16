# 24 place solution

Competition: ariel-data-challenge-2025
Rank: #24
Source: https://www.kaggle.com/c/ariel-data-challenge-2025/writeups/24-place-solution

Repository: https://github.com/vilka-lab/Ariel-Data-Challenge-2025/tree/master
Notebook with submit: https://www.kaggle.com/code/ivanilyushchenko/24-place-solution

The solution to the problem is shown in the diagram.



1. Using an open solution via polyfit with correction via a neural network gives much better results than directly predicting the value (approximately +0.1 to the metric).
2. I tried two losses - MSE and direct loss through the competition metric. MSE does not allow predicting uncertainty, but the competition metric is quite unstable. I came up with a complex metric with weights of 0.9 for MSE(*1e6) and 0.1 for the GaussianLogLikelihoodLoss. The implementation of a loss is here: https://github.com/vilka-lab/Ariel-Data-Challenge-2025/blob/master/src/loss.py
3. Channel-wise normalization gives a large increase in metric. Normalization to the white curve worsens the result (~0.03).
4. There is too little data for this solution. The network fit in 500 epochs with a batch of 32, after 300 epochs it goes into overfit. If you divide the data into 5 folds, the average score is 0.01-0.02 worse than 10 folds, and the overfit occurs earlier. Most likely, metric can be significantly improved with fit the network on a larger dataset.
5. Considering point 4, I spent quite a lot of time trying to generate additional data using exosim2. Unfortunately, it didn't work out for me - the generated data was shifted relative to the competition data and worsened metrics.
6. Attempts to find the best backbone for 2d features did not bring results - vit_base_patch32_224.sam_in1k turned out to be the best. A further increase in the size of the network simply accelerates the moment of overfit.
7. Augmentations help to overcome overfit somewhat, and most likely a solution can be found here that significantly increases the metrics. I couldn't do it.
8. Attempts to manually adjust the result for bad samples with transit curves did not bring results - the model from each fold may work differently with such samples. Some models worked well with bad transits and poorly with ideal ones. I added a feature that indicates whether it's a good sample or not, and this raised the metrics by ~0.015. A good solution here is to simply generate more transit curves and/or make the appropriate augmentations, but see 5 and 7. Well, or I just did something wrong.
9. Preprocessing from competition hosts does not affect the result, so i simply remove it and speed up calculations. Attempts to use different prefilters also failed.
10. Solution works fast. The final submission contains 20 models - 10 best and 10 last from 10-fold CV and it completes in about 2 hours.

Thanks for your attention.
