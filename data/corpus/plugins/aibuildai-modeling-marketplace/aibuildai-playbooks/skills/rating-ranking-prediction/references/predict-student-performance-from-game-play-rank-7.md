# 7th Place Solution (Efficiency 1st)

Competition: predict-student-performance-from-game-play
Rank: #7
Source: https://www.kaggle.com/c/predict-student-performance-from-game-play/discussion/420119

I am pleased to have fought the long and hard competition with all of you here.
Here I would like to outline my solution.

## Overview

- To make predictions for 18 questions, I trained 3 LightGBM models, one for each level_group. The reason I did not build a model for each question was primarily to reduce inference time.
- Most of the features I have created are features based on the time difference between two consecutive actions. (More on this later.)
- The CV score was improved by about 0.002 by adding raw data published by the competition host.
- Unexpectedly, the submission for the Efficiency Prize had the best score in Private Leaderboard amoung the selected sumissions. The inference time of that is approximately 3 minutes.

The notebooks reproducing my submission are as follows:
- https://www.kaggle.com/code/rsakata/psp-1-save-data
- https://www.kaggle.com/code/rsakata/psp-2-process-raw-data
- https://www.kaggle.com/code/rsakata/psp-3-fe-and-train-lgb
- https://www.kaggle.com/code/rsakata/psp-4-test-inference

## Feature Engineering

The six variables (level, name, event_name, room_fqid, fqid, and text) were concatenated as aggregation keys, and the time difference from the previous or following record was summed for each key and used as the feature. If written in pandas-like code, 
`df.groupby(['level', 'name', 'event_name', 'room_fqid', 'fqid', 'text'])['elapsed_time_diff'].sum()`

In addition to the time difference from the previous or following records, the number of occurrences of each key is also added as a feature. Since these features can be calculated by sequentially reading the user's session, they can be calculated very efficiently by treating the data as the Python list instead of using Pandas.

Furthermore, the record whose event_name is 'notification_click' is considered as a important event, and the time difference between the two events is added to the feature.

The procedures for calculating these features can be found by reading the third published notebook.

## Modeling

Since the variety of keys (combinations of six variables) is very large, I reduced features before training by excluding in advance rare combinations that appear only in a small number of sessions. However, since the number of features still amounted to several thousand, I first trained LightGBM with a large learning rate (0.1) and performed feature selection based on gain feature importance. The training was then performed again with a smaller learning rate (0.02) using 500 to 700 features.

In the second modeling, raw data published by the host (https://fielddaylab.wisc.edu/opengamedata/) was included in the training. Although I was unable to reproduce the host's train.csv file completely, but I was able to reproduce it approximately using the second published notebook.

Many of the sessions included in this data were different in nature from the competition data because they did not complete the game until the end. In fact, users who left the game midway through tended to have lower percentages of correct responses. To reflect this difference, the maximum level of each session was added as a feature.

When training the model for the last level_group, I augmented the label of the second level_group, which contributed to the improvement in accuracy. I believe that the reason for this is that overfitting was suppressed by using more data to determine the split point when splitting nodes of decision trees. However, for the first and second level_groups, this data augmentation method did not contribute to improve accuracy in local validation.

The CV/LB scores of my best submission is:
- CV: 0.7034
- Public LB: 0.703
- Private LB: 0.703

## Other Remarks

- For stability of evaluation, 4-fold CV was repeated three times with different seeds.
- Based on the validation results, the threshold was set at 0.625. No adjustment was made for each question.
- To reduce inference time, models trained in CV were not used, but retrained models using all data were used for inference.
