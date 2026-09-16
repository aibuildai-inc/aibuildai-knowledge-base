# 1st place solution

Competition: nfl-impact-detection
Rank: #1
Source: https://www.kaggle.com/c/nfl-impact-detection/discussion/209403

Thanks to the organisers of this competition, it was a very interesting task to solve!

My approach was very similar to the other top solutions: 2d detection of all helmets, tracking of helmets, 3d classification of crops and post processing to suppress false positives.

# Detection

I used the YoloV5-l trained on 10k images using the full resolution.
I started using EfficientDet based detector with modified set of anchors to better match the helmet size distribution and trained on video frames, but YoloV5 worked better.

# Helmets Tracking

For each detected helmet, I found the optical flow between a few surrounding frames.
I tried to use either OpenCV or RAFT (Thanks @daigohirooka for the great notebook and referring RAFT). I used the optical flow to track helmets between frames and to estimate the average helmet velocity on the image plane over a few surrounding frames. I think RAFT worked slightly better but my top submit still used OpenCV for helmets tracking.

# 2.5D Classification

For classification I used 16x3x128x128 crops, scaled so the median helmet size over the current frame is mapped to the center 40 pixels. It provides the consistent scale, robust to detected bounding box variations, especially for partially visible helmets.



I also corrected for the linear helmet movement between frames. The current frame at 8th slice is always centered, but all the other frames are shifted using the current box velocity, estimated during the tracking stage. So when the player is running with the constant speed or camera is panning, the helmet stays at the frame center, but during acceleration it would move to and from center.

The intuition behind - the acceleration is important for classification, but it's harder to estimate on top of potentially fast movement due to camera movement.

For classification models, instead of 3d convolution I used the [Temporal Shift Module](https://arxiv.org/pdf/1811.08383.pdf):

[Temporal Shift Module]

TSM has very little overhead comparing to N 2d models, can reuse any ImageNet pretrained 2d classification model and performed reasonably well for video understanding tasks.

I also had promising results with the [MotionSqueeze: Neural Motion Feature Learning for Video Understanding](https://arxiv.org/abs/2007.09933) approach, which adds flow estimation to TSM based model, but the implementation is not trivial and does not have license specified.

I found small models with TSM added at the start of residual blocks work well and are usually easy and fast to train (from an hour to a few hours per model). I used the ensemble of EfficientNet B0-B3, Resnet18 and Resnet34. When comparing the TSM approach to 3d convolution, it's easy to notice the lack of pooling in the time dimension, so instead of shifting features by one frame, I shifted by 2 and 3 frames in the last blocks of resnets/efficient nets to mimic the dilated convolution.

During training I marked 3 frames around the impact as positive and used 5 or 10% positive samples.
In addition to annotated training labels, I added the false positive prediction from a few undertrained detection models. I used 4 folds CV, for submission I averaged predictions of multiple models from all 4 
 folds. To speed up prediction, I ran classification on a few fast models first, to filter 95% of simple to predict negative samples.

# Post-Processing

The postprocessing is quite simple, I select the detected/classified box with highest impact confidence and use flow/tracking information to suppress the detections for the same player for surrounding frames, with suppression decaying from the current confidence to 0 over 16 frames. I used the same threshold for all video files and frames.

I checked if such post-processing would suppress the positive samples using training dataset and found surprisingly significant amount of positive samples within 3-10 frames for the same player. Most of such sequential impacts were of the different type, for example helmet to helmet impact followed by helmet to body. To address this, I changed the classification model to predict the impact type as well, and suppressed the predicted impact type only. The score improvement was quite small, less than 0.01 estimated on local CV.
