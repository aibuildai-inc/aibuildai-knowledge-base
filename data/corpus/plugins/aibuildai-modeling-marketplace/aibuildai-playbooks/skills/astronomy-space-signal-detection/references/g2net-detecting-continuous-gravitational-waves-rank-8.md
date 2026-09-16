# 8th place solution: matched filter and CNN

Competition: g2net-detecting-continuous-gravitational-waves
Rank: #8
Source: https://www.kaggle.com/c/g2net-detecting-continuous-gravitational-waves/discussion/376253

Congratulations to the winners, and I would like to express my gratitude to the organizers and my teammate.

# Summary
* Matched filter using PyFstat
* CNN with pseudo-label and synthesized data
* Ensemble MF and CNN results.

# Matched filter
Matched filter (MF) is a highly competitive method for searching continuous waves. PyFstat provides a MF module, which we used for this competition. However, PyFstat (and its underlying LALpulsar) requires a specific file structure (.sft file) that is not compatible with hdf5 or ndarray. We converted the provided data to the sft file.  
When provided with four values: f0, f1, alpha, and delta, matched filter calculates a "twoF" value, which is the statistical measure of "likelihood" of the target wave's existence, 

* We used PyFstat's [SemiCoherentSearch](https://pyfstat.readthedocs.io/en/latest/pyfstat.html#pyfstat.core.SemiCoherentSearch) with "nsegs=1000".  
  Here, we used relatively large "nsegs" value to smooth the function shape of "twoF" w.r.t. parameters f0/f1/alpha/delta. This trick helps find the existence of target waves in much less grid search trials.
* We conducted a grid search for alpha and delta.
* We used Optuna to search for f0 and f1 in order to maximize the towF value.
* We performed around 600 Optuna explorations per sample.
* You can find more information about our implementation here:   
  https://www.kaggle.com/code/iiyamaiiyama/g2net-pyfstat-matched-filter

## CV result
The matched filter histograms for the training data (600 samples) are shown. The right image is an enlarged version of the left image. As you can see, if MF result(twoF) is greater than 4600, the precision is 100%. Therefore we set 4600 as the threshold value.



The AUC of the 600 samples is 0.8406.  


## LB result
If we submitted only the MF results, the private/public leaderboard score was 0.764/0.760, which is not a very competitive result. We believe it may be due to the non-stationary noise. Therefore, we decided to ensemble MF and CNN results to improve our score.

# CNN
## dataset
The training data only contains 600 samples. We had to deal with it.

### Stationary noise
* Pseudo-label using matched filter
We selected samples with MF results greater than 4600 from the test set. Around 1700 samples.
* Synthesized data
We synthesized new data from train1 and train2 by using the following formula:  
  `new_data = train1 * alpha + train2 * beta`  
  (alpha and beta satisfy the condition `alpha^2 + beta^2 == 1`)

### Non-stationary noise
We used the following notebook as a reference for creating our own new data.   
https://www.kaggle.com/code/vslaykovsky/g2net-pytorch-generated-realistic-noise  
After creating the new data, we augmented the data by synthesizing the two data as well as the stationary noise data.

## Model
We had two separate models: one for stationary noise and one for non-stationary noise. The models themselves were ordinary CNNs. We used data augmentation techniques such as h/v flip, mask and roll.

# Ensemble
Finally, we ensemble MF and CNN results. For samples with MF result greater than 4600, we assigned a score of 0.9-1.0. For all other samples, we used the CNN scores of 0.0-0.9. By Prioritizing samples with 100% precision and placing less confident CNN results afterwards, we can maximize the AUC score.
