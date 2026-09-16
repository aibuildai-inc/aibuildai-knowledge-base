# 1st Place Solution

Competition: nfl-health-and-safety-helmet-assignment
Rank: #1
Source: https://www.kaggle.com/c/nfl-health-and-safety-helmet-assignment/discussion/284975

Thanks to organizers for this interesting challenge and congrats everyone who enjoyed it! It’s one of the most fun competitions that I’ve ever experienced. I especially appreciate Rob’s supports before and during competition. I believe he makes this competition such great thing.

It’s really tough challenge. It requires various skills of  detection, registration, optimization and tracking (and debugging). I guess many competitors build complicated pipelines, and  struggled to debug it. I respect those who have completed this competition until the end. Congratulations!

# Overview of my solution
My pipeline consists of:
- **Detector to find helmets.**
- **Converter to project movies’ images to 2D map (bird’s eye view).**
- **Classifier to classify players into 2(H/V) teams.**
- **Registration of detected players on 2D map to provided tracking data.**
- **Track detected bounding boxes and reassign players.**

I guess the **key difference between my solution and public notebooks is the mapping and registration modules**, which gave me the score of approx. 0.8 only by helmets.csv though it works faster than 10 frames/sec on GPU.
Let me explain each modules briefly.
[pipeline]


## 01. Helmet Detector
- 2 stage detector to find helmets.
- 1st detector predicts the average helmets’ size and resize input images(higher resolution) based on it. 
- 2nd detector detects helmets in high resolution images. I thought detecting fixed-size objects is easier than detecting objects of various sizes.
[detector]

## 02. Image to Map Converter
- CNN(U-Net base) to convert helmets’ bounding boxes in images to 2D map. It predicts 2D location(x and y) looking from camera location.
- It outputs the global location from bottleneck and small residuals from decoder.
- Attention by helmets bounding-boxes improves accuracy.
[image2map]

## 03. Points to Points Registration
- Matching predicted players on 2D map to the provided tracking data.
- ICP (Iterative Closest Points) based algorism. Iteratively solve the nearest search and normal equation to get 4 unknown parameters (xy translation, rotation and zoom ratio) by least squares fitting.
- Pre/post-processing is applied to remove inappropriate predictions such as sideline players.
[p2p_registration]
[pre_registration]

## 04. Team Classifier
- Team information is also important to improve the registration accuracy. X,Y location and team can be used in Points to Points Registration above.
- CNN classifier predicts the similarity matrix to show each pair of players belongs to the same team or not.
- Using arcface and pseudo-labeling improved the accuracy. Validation score is approx. 97%.
[team_classifier]

## 05. Tracker
- Tracker accumulates the results of player assignment through all frames, and re-assign players to bounding boxes.
- I applied simple IoU tracker because it’s fast and enough accurate.
- Not only frequency but also tracking confidence and frame distance is used for reassignment. Assignment results far from the target frame should not be weighted.
[tracker]

## 06. Ensemble
- WBF(Weighted Box Fusion) is applied in the reassignment phase in the tracker. It ensembles multi-frame and multi-model predictions.
- Take the weighted average of the player-assignment matrix of each models, then choose the final assignment by Hungarian algorithm. It works better than normal WBF which ensembles the results of bounding boxes and detection confidences only.
- 4 detectors’ ensemble for final submission. 
[ensemble]

## Scores History
[scores_comparison]

# Final thoughts
Fortunately, I have the experience of the object detection, depth estimation, pointcloud registration and tracking. I think this is the reason why I could win this competition.
I started learning programming and machine learning at kaggle 3 years ago, and most of my knowledge comes from kaggle. Thanks again to the whole of kaggle community!


### Inference code is released
https://www.kaggle.com/kmat2019/nfl-1stplace-inference

## Ablation Study (Late Submission)
[scores_comparison]
