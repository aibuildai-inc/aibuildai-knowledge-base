# 4th place solution

Competition: nfl-impact-detection
Rank: #4
Source: https://www.kaggle.com/c/nfl-impact-detection/discussion/208947

When I started solving this competition, a solution based on a two-class detector and post-processing was already published on the forum. At first I wanted to use this approach, but after looking at the data carefully, I realized that in order to achieve a high score, it is necessary to use a temporal context. Therefore, I chose a two-stage approach: helmet detection and ROI classification using a 3D convolutional network.

## Data preparation
For convenience, I extracted all video frames into separate files. There weren't very many frames, so I saved them to the SSD without any problems. Then I split all the data into training and validation (20%) randomly by game_id.

## Helmet detector
It was the easiest part. I trained Faster-RCNN with ResNet-50 FPN backbone using [detectron2](https://github.com/facebookresearch/detectron2). I made the following changes to the [standard](https://github.com/facebookresearch/detectron2/blob/master/configs/COCO-Detection/faster_rcnn_R_50_FPN_1x.yaml) config:
- reduced the minimum anchor size from 32 to 16, since there are many small helmets in the dataset
- used single class (helmet only)
- used both frames and images for training
- used SyncBN  

The first trained model achieved a fairly good detection accuracy. On validation, the AP50 is 93.607 (the recall of ground truth boxes with definite impact label is 84%). I did not do more experiments with the detector, but perhaps a more accurate detector would give a better final result.

## ROI classifier
I used FAIR's [SlowFast](https://github.com/facebookresearch/SlowFast) library to train the ROI classifier. With this library, I have implemented two alternative approaches:
### Action detection
The approach is as follows: a sequence of full frames is fed to the 3D CNN input, a feature map is calculated, and using the ROIAlign operation, features for the ROIs are extracted and classified. I used the SlowFast model with the ResNet50 backbone with the following modifications:
- used a sequence of 20 frames
- removed stage5 from the backbone and added 3 upscale deconvolutions to make the feature map large enough
- for each ground truth impact added additional ground truths to +-2 adjacent frames
- increased the number of frames that have an impact to 25% of the total number of frames using oversampling
- from augmentations used multiscale, random crop, random flip and color augmentation
- to remove duplicates on inference, I used an IOU tracker. The tracker combines impact boxes from adjacent frames and then the most confident prediction is selected from each track.  

The result of this model is 0.3915 on the private leaderboard.

### Action classification
The approach is as follows: the sequence of frames cropped around the target box, then the sequence of crops is fed to the 3D CNN input, and then the impact probability is calculated directly. Again I used the SlowFast model with the ResNet50 backbone with the following modifications:
- used a sequence of 20 frames
- used 128 x 128 input size
- used 5 input channels instead of 3 RGB channels. The first additional channel is the heatmap of the center of the helmet of interest. The second additional channel is the heatmap of the centers of all helmets (see picture below).
- for each ground truth impact added additional ground truths to +-2 adjacent frames
- increased the number of definite impact crops to 10% of the total number of crops using oversampling
- from augmentations used multiscale, random crop, random flip and color augmentation
- to remove duplicates on inference, I used an IOU tracker. The tracker combines impact boxes from adjacent frames and then the most confident prediction is selected from each track.  

[input example]

The result of this model is 0.6037 on the private leaderboard. Then I added horizontal flips as a TTA and increased the score to 0.6341. Finally, I added the action detector described above to the ensemble and got a minor improvement: 0.6355 on the private leaderboard.
