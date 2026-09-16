# 14th Place Solution

Competition: child-mind-institute-detect-sleep-states
Rank: #14
Source: https://www.kaggle.com/c/child-mind-institute-detect-sleep-states/discussion/460274

This competition was a great learning experience for me. I would like to thank the organizers!

## Overview

- The neural network was trained to directly predict the timing of events.
- The training was performed as a regression task, with 'wakeup' set to 1 and 'onset' set to -1 and decayed before and after the occurrence of those events as the target.
- The neural network structure is a modified version of U-Net with 1D-CNN.
- Submission file was created by extracting peaks for the average of the model output in the 10-fold CV.
- Post-processing before peak extraction improved the CV score and Public LB Score, but had little effect on the Private LB Score.
- The public/private scores of my final submission are 0.755/0.821 without post-processing, and 0.770/0.824 with post-processing.

The notebooks of my solution are as follows:
- https://www.kaggle.com/code/rsakata/cmi-14th-place-solution-train
- https://www.kaggle.com/code/rsakata/cmi-14th-place-solution-inference

The public/private scores of the above notebook are 0.748/0.822 without post-processing, and 0.768/0.822 with post-processing. The differences appear to be due to minor changes made in the process of refactoring the code or simply due to randomness. (I consider it not essential.)

## Input

The data for each series_id was divided into daily segments and input to the neural network. To be precise, however, in order to avoid the influence of padding in the CNN, the data of the previous and following days were partially combined and input.

The number of input channels to the neural network is three, as follows:
- logarithm of the 2-minute moving standard deviation of 'angelz'
- logarithm of 'enmo'
- flag to identify dummy data

Like the others, dummy data was detected by duplicating (anglez, enmo, time) in the same series_id. Predictions for dummy data were set to 0 before calculating the loss, so as not to affect backpropagation. (Therefore, the third input is not that important.)

Features related to time were not added to the input to the neural network, and all temporal trends were considered in post-processing.

## Target

In the evaluation metrics of this competition, scores can be obtained even if the timing of event occurrence detection is slightly off, so the target should reflect not only the moment of event occurrence but also its surroundings when training.
On the other hand, however, insufficient target decay tended to make the timing of peaks only roughly predictable, leading to worse scores at smaller thresholds. After searching for the method that would yield the best validation score, I finally adopted an exponentially decaying form, as shown in the figure below.


## Model Architecture

The architecture of my neural network is based on the following notebook of the other competition written by K_mat (2nd place in this competition).
https://www.kaggle.com/code/kmat2019/u-net-1d-cnn-with-keras

The input granularity is every 5 seconds, but the output granularity is every minute, which differs from regular U-Net. I tried various structures in terms of number of layers, number of channels, kernel size, etc., but the search was limited and I believe that better configurations exist.
For details on the structure of the model, please see the notebook.

## Post-processing

No time information was input to the neural network, which was taken into account in post-processing. The minute-by-minute scores output by the neural network were multiplied by weights according to the trend by time of day.
Specifically, they are as follows:
- probability of each events by time of day across all series_id
- probability of each events by 'minute mod 15'
- average of scores by time of day for each series_id (to capture periodicity at the series_id level)

In addition, as a post-processing step unrelated to time, the scores were multiplied by a coefficient based on the percentage of dummy data. This is because the series_id with more dummy data tended to be somewhat more difficult to guess the peak accurately, and I aimed to improve PR-AUC by placing more confident prediction at the top of the list.

Peak extraction using the `scipy.signals.find_peaks` function was performed on the adjusted scores. A two-step strategy was used, first extracting the larger peaks and then the smaller peaks. The former extracts approximately one peak per day, while the latter is countless. By amplifying the former scores, I aimed to improve the PR-AUC.

These post-processing improved the CV score by about 0.005 to 0.01, but unfortunately, as mentioned earlier, the impact of these post-processing on the Private LB Score was limited.
