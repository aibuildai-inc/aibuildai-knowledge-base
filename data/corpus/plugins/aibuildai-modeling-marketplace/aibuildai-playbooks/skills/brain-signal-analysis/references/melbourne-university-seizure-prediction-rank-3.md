# Solutions (?)

Competition: melbourne-university-seizure-prediction
Rank: #3
Source: https://www.kaggle.com/c/melbourne-university-seizure-prediction/discussion/26039

I really enjoyed this competition, it's a very worthwhile problem to try to solve and I hope it yields a few ideas that work out in the real world (rather than just how to optimally exploit a data leak!)

It's probably a bit early to talk about good solutions given the likely scoreboard shakeup ([be scared](https://www.kaggle.com/c/santander-customer-satisfaction/leaderboard)), but I'm really interested to know how others have approached this problem, regardless of if it seems to work well or not - and particularly if they went with single-subject models or general models. Also we can place bets on which approaches might do well or not on the private LB.

For what they're worth, I'll briefly outline my approaches to processing, fitting, and validation below, and flesh it out a bit later if it doesn't entirely tank in the shakeup. Criticisms, ideas, and speculations on how it'll do on the remaining 70% are very welcome.


### Software
 - MATLAB 2016b

### Preprocessing
- I added the leaked preictal data from the original test set to the training set. This data didn't contain sequence information and needed to be treated slightly differently to the original training data.  

- I didn't try to recreate any sequence information in the training or test set that may have been revealed by the leak.
- Prior to feature extraction, the 60-minute blocks of the original training data (6 consecutive 10 min files) were concatenated and then split in to epochs of between 100 and 600s. Single files were epoched individually.
- No additional filtering, de-noising, normalizing, or referencing of raw data before feature extraction.

### Feature extraction 
- Features:
   - Frequency power of [bands](https://en.wikipedia.org/wiki/Electroencephalography) on each channel
   - Summary statistics of each channel (mean, RMS, std, kurtosis etc.)
   - Correlation between all channel pairs in temporal and frequency domains 
   - Averaged versions of above (across channels)
- The final training/test sets was created by joining the features extracted from different combinations of epoch window lengths.

### Fitting & cross validation
- I used general models trained on all 3 subject's data for the actual submission, not single models (which scored at best ~0.64 on the new test set). 
- Cross validation was a pain and I had to code a new CV object to handle it in MATLAB.  I used a variation of the CV approaches discussed on the forums, It'll be interesting to see how well local CV holds up for the final scoreboard: 
	- K-fold validation where any files in the same group of 6 were included in the same fold. Single files were as a "group" containing 1 file.
	- If all files were treated individually, local CV scores were between 0.82-0.92 and didn't correlate much with leaderboard score. 
- I fit two types of model with no hyperparameter tuning: 
	 - [Polynomial SVMs (quadratic)](https://en.wikipedia.org/wiki/Polynomial_kernel), (performed well in early tests), local CV scores:
       - s1 single model: ~0.77±0.03
       - s2 single model: ~0.79±0.02
       - s3 single model: ~0.08±0.02
       - General model: ~0.77±0.01
	 - [RUS boosted tree](http://sci2s.ugr.es/keel/pdf/algorithm/articulo/2010-IEEE%20TSMCpartA-RUSBoost%20A%20Hybrid%20Approach%20to%20Alleviating%20Class%20Imbalance.pdf) ensembles (designed to handle imbalanced data sets), local CV scores:
       - s1 single model: ~0.74±0.01
       - s2 single model: ~0.75±0.02
       - s3 single model: ~0.78±0.02
       - General model: ~0.75±0.01
	- The predictions of two general models were averaged giving a public LB score of roughly 0.775 - 0.815. The combined predictions were reliably better than either model alone.


### Concerns
 - During the last weeks of the competition, minor changes resulted in small changes to local CV scores, but LB deviations of ±0.03. 
 - My general model predicted subject 1 was the most likely to have a seizure. The training set includes a greater proportion of seizures for subject 1 (30% compared to 7% for all subjects before the leak and ~10% of subjects 1 and 2 after the leak). Is the model relying too much on subject 1? How much of subjects 1's seizure data is left in the remaining 70% of the test data?
 - The predictions of my single models scored similarly to the general models in CV (usually better), but poorly on the LB. Also plots of the predictions looked different to the general models. The single models also scored better than the general models on the first training/test set. I'm not sure why general models worked better with the second test set.

Good luck everyone!
