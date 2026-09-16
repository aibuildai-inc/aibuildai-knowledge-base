# 3rd Place Solution -- Simple 2-Stage Pipeline

Competition: nfl-impact-detection
Rank: #3
Source: https://www.kaggle.com/c/nfl-impact-detection/discussion/208787

Thanks to the organizers and congrats to all the winners and my wonderful teammates! @boliu0 @garybios

The dataset of this competition is much more complicated than normal ones and full of places to explore.

So I'd like to call it Multi-view Video Event Detection task, rather than simple Object Detection task, because:

* This data set contains time series information
* This data set has multi-view for a same scene
* Although the impact bbox is marked on the helmet, the marking must depend on the environment around the helmet

In addition, the understanding of the metric is also extremely important for this competition.

# Summary

* Only use videos and images.
* Use EfficientDet to generate candidate impact bboxes, crop them into small images.
* Use binary image classfication model to classify helmet crops through 9 frames.
* Adjust scores for each bbox by multi-view information (Post-processing).
* Drop similar bbox through 9 frames (Post-processing).

This is a illustration of our pipeline (except for post-processing):



# Generate Candidate BBoxes

We train EfficientDet on 2 classes (helmet and impact) then take all bboxes with class == 'impact' and score > 0.17 as candidate bboxes.

If a player has an impact, we regard all his bboxes within +-4 frames as positive bboxes, because the metric treats predictions within 4 frames as TP. Base on this rule, if a positive bbox appears in a frame, we will regard it as a positive frame

While training EfficientDet, we use all positive frames and 50% of the negative frames (which mean no positive bbox in it) from all videos, and all the 10,000 static images.

# Binary Classification Model

As shown in the illustration, we use binary classification models to further predict whether bboxes crops are impacts or not.

For training binary models, we expand the height and width of the bbox in the videos to 3 times the original length, crop them from frames, and use this as the training data for the binary classification model.

When cropping a specific bbox, we crop it through 9 frames (t-4, t-3, t-2, t-1, t, t+1, t+2, t+3, t+4), turn it into gray scale and stack together to get an input of shape (h, w, 9)

It should be noted that the `9` here is a variable parameter.

Through this method we can get around 900k helmet crops, which is consistent with the number of bboxes in this data set.

But unlike training EfficientDet, when training binary models, we only regard the impact bbox and its +-1 frame corresponding bbox as positive, which is less than 6k.

When predicting, we use the same cropping method for all the candidate bboxes and then infer these crops by binary models.

Through the binary model, our single model local score jump from 0.3x to near 0.6 with our post-processing methods, which is the biggest highlight in our pipeline.

# Multi-View Post-Processing

We found that when one view has an impact, there is a high probability that the other view has an impact within the +-1 frame range. Therefore, we tuned separated thresholds depending on the predictions in the other view. For instance, the threshold for a certain Endzone frame depends on whether there is a predicted bbox in the Sideline view within +-1 frame. If yes, the threshold is lower (say 0.25); if not, the threshold is higher (say 0.45).

# Drop Similar BBox

According to the official metric, although the predicted bbox is counted as TP in the +-4 frame range, a label can only accept one predicted bbox. Therefore, we hope that an impact will only output one prediction bbox.

So we designed a function to drop all but one "similar" bboxes within any 9 consecutive frames, keeping only the one with largest confidence. For this purpose, we define any cross-frame bbox to be similar when they have iou > 0.25

# Ensemble

Finally we ensemble 7 EfficientDet Models and 18 Binary Models that achieved local score 0.64, public LB 0.66 and private LB 0.69

# Acknowledgement

Special Thanks to Z by HP & NVIDIA for sponsoring me a Z8G4 Workstation with dual RTX6000 GPU and a ZBook with RTX5000 GPU.

It's really fast and powerful enough for this harsh competition!
