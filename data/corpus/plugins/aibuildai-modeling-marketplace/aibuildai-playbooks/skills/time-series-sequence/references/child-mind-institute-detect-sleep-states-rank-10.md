# 10th Place Solution

Competition: child-mind-institute-detect-sleep-states
Rank: #10
Source: https://www.kaggle.com/c/child-mind-institute-detect-sleep-states/discussion/459894

First and foremost, I would like to express my gratitude to all participants and organizers.
It was a very tough competition as I did not have many good scores on public LB, but I was satisfied with the final results.

## Features engineering
+ enmo
  + Utilized as is
+ anglez
  + Used `anglez.diff().abs()`
+ Time Encoding
  + Normalized the hours into t=[0~1) and employed `sin(2*pi*t)` and `cos(2*pi*t)` as features
  + Experimented with finer periodicities (e.g., `sin(4*pi*t)`, `sin(8*pi*t)`, `cos(4*pi*t)`, `cos(8*pi*t)`), but performance worsened
  + Considered features like week and month, but they were rejected due to performance degradation.
+ Duplicate Data
  + Identified instances of identical sensor data occurring in the same series at the same time.
  + Assigned a flag of 1 for times with duplicates and 0 for times without duplicates
+ Step
  + Flags set for within 1 hour and within 8 hours from the start of measurement
  + Because, included data points where onset occurred within 1 hour or wakeup within 8 hours since there were none

## Model
Adopted a 1D-UNet GRU model as illustrated. Features, excluding enmo and anglez, are added just before the GRU.  Trained using Binary Cross Entropy.


 


## Target Design
Utilized a heatmap based on a Gaussian distribution. For stability in training and improved AP at a large tolerance, a Gaussian distribution with a large sigma was advantageous. Contrastly, a Gaussian distribution with a small sigma led to instability in training but improved AP at a small tolerance. As a compromise, a weighted sum of Gaussian distributions with large and small sigmas was used.



```python
gauss_small_sigma_minute = 2.5
gauss_large_sigma_size = 10
gauss_mix_ratio = 8
gauss_small_sigma_size = minute2step(gauss_small_sigma_minute)
gauss_large_sigma_size = minute2step(gauss_large_sigma_minute)

gauss_large = np.exp(-((x - gauss_center)**2) / (2 * (gauss_large_sigma_size)**2))
gauss_small = np.exp(-((x - gauss_center)**2) / (2 * (gauss_small_sigma_size)**2))

gauss = (gauss_large + gauss_mix_ratio * gauss_small)
gauss /= gauss.max()
````


## Training
+ Randomly sample 12 hours
+ 150 epoch, AdamW
+ Apply SWA after 50 epoch

## Inference
+ In contrast to training, infer a entire series at a time

## Ensemble
The processing time for the entire test dataset, including data loading, pre-processing, and post-processing, was approximately 18 minutes.
Of that time, model inference took 45 seconds.

Public LB is calculated by 25% of the test data, so some shake was expected.
Therefore, the decision was made to perform ensemble learning with large number of models. The final ensemble consists of 120 models, and the processing is completed within 120 minutes.

## Post-processing
It is crucial to detect multiple candidates in one night.

+ Smooth the output values with a width of 11
+ Detect all local maxima with a threshold of 0.01 and add them as candidate points
+ For all remaining timestamps, perform the following in descending order of output values
   + If the timestamp is more than 21 minutes away from any previously detected timestamp, add the product of 0.1 and the output value as a candidate point
