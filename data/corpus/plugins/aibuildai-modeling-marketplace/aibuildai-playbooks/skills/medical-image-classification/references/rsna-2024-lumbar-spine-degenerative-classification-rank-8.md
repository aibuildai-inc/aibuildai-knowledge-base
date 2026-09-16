# 8th Place Solution

Competition: rsna-2024-lumbar-spine-degenerative-classification
Rank: #8
Source: https://www.kaggle.com/c/rsna-2024-lumbar-spine-degenerative-classification/discussion/539548

Thanks to RSNA and Kaggle team for hosting such interesting challenge. Thanks @stgkrtua for teaming up with me. This was my first time participating in a medical competition, but I enjoyed it very much.

## Overview
Our core architecture of our solution is shown in the following figure. It consists of 2D classifier and 1D classifier.



### 2D classifier:
- Takes a 2-dimensional image as input.
- Predicts 25x3 severity labels (=5 conditions x 5 levels x 3 class).

### 1D classifier:
- Stack the output of the 2D classifier.
- The final prediction is predicted by considering the 1st-stage predictions for every instance ID.

One of the uniqueness of our model is the **feature extraction** of 1st stage.
Instead of cropping out the important parts of an image, we trained our model to extract the features corresponding to these regions. So we trained the detector as a sub task and used the heatmap as the weight of the feature extraction like an attention.

This model provides several advantages, 
- it enables to consider the overall **context of the image**.
- it gives robust outputs that are **less sensitive to ambiguities in detection**.

## Other findings
### SSL
Self-supervised learning slightly improved the public/private score. The model is trained to learn the multi-view similality by using the xyz coordinate. Unfortunately, CV score didn't change. I guess the labels of test dataset are cleaner than train dataset.


Red area in the figure shows the high similatiry area between two images.

### Relabeling
As you know, there were many inconsistencies in the labels. 
I created a super cool annotation tool and tried to correct the labels myself, but I couldn't do it well due to the lack of domain knowledge. If anyone has made corrections, I would appreciate it if they could share the results.


**Creating the annotation tool was one of the most enjoyable parts** of the competition, even though it didn't end up being very useful.

## Code
The inference code is available at
https://www.kaggle.com/kmat2019/rsna8th-inference

Due to a minor bug, the score is slightly better than the final submission.
