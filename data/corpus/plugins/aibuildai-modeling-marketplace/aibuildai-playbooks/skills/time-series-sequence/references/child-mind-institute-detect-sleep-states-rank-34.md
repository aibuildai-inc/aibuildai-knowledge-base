# 34th Place solution

Competition: child-mind-institute-detect-sleep-states
Rank: #34
Source: https://www.kaggle.com/c/child-mind-institute-detect-sleep-states/discussion/459595

First of all, I would like to thank the organizers and kaggle staff for organizing the competition. I learned a lot through the competition.
# Summary

- based on 213tubo's repository
- Data cleansing
- Ensemble models
- Reduce FP by post-processing

# data cleansing

When I visualized `train_series.parquet` and `train_events.csv` by series_id as shown below, I noticed that the label data was dirty.

Labels were not assigned where they should have been, or were assigned where they should not have been (where the organizer may have filled in values for missing values).

There were many areas of concern, such as labels that were assigned at times that were probably tens of minutes or more off from the time the subjects were onset/wakeup.

We thought that such dirty label data would be an obstacle in training the model, so we examined the data for every series_id and cleansed the label data by shifting or adding the position of the labels.





# Model
- Unet
- encoder
  - maxvit_tiny
  - eca_nfnet_l1
  - dm_nfnet_f0
- Augment
  - Mixup
  - Spec_augment
- Label_smoosing(sleep)
- Sliding inference
- Not using both ends of the inference

# Post-processing

When I visualized the data, I noticed that some of the data showed unnatural periodic patterns, such as `1e6717d93c1d`.

I considered these unnatural periodic patterns to be invalid because the organizers had filled in common values for each time of the day for periods when no data were measured.

The model may output onset/wakeup predictions within these periodic patterns, but they are considered false positives.

We post-processed the model to remove predictions within these periodic patterns.

The criteria for periodic patterns are as follows,

- When anglez and envo are shifted back and forth 17280*n steps, if both anglez and envo values are consistent, the prediction is considered invalid.
- However, if they do not match for two or more consecutive steps, they are not judged invalid (considering those due to coincidence)


