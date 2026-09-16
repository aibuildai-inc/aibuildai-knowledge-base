# 27th place solution

Competition: deepfake-detection-challenge
Rank: #27
Source: https://www.kaggle.com/c/deepfake-detection-challenge/discussion/145965

## Approach

We used a simple frame-by-frame classification approach.

## Face-Detector

We used simple MTCNN detector.
Input size for the face detector was scaled down by 50% to significantly reduce inference time and memory usage without much loss in detection accuracy. We then used the bounding box to extract a square shaped face with image center coordinates and height being the same, but adjusted width.

## Final Ensemble
Our final ensemble consists of B5/B6/B7 single model ensemble.
 @ratthachat will give further details on model training in the comment section.

## Models
We mostly used EfficientNets as we did not have much success with others Models.
(1 Xception was used in the worse performing final essemble)
We played around with input sizes and sticked with 224*224 as most cropped faces in the training dataset were around that size.

We started with B0 - B2 with some success. 
But our final solutions of B5-B7 outperformed them.

## Memory Usage and Inference Time

Due to memory constraints we used a batch generator that extracts and predicts only on two frames from a video at once. This also enabled us to use parallel inference to significantly reduce run-time and predict on every 27th frame in the end.

## Things that may have worked

We considered organizers willl introduce organic videos and different deepfake methods in the test set.
Thats why we constantly evaluated our models on organic videos using Predictions and GradCam and used face crops without big margin, we thought this may generalise better. We used a margin of 14 pixels from the MTCNN face detector. 

The final prediction was a simple average of all predictions per video. We had better results splitting predictions per actor, but due to time reasons and as we considered organizers introducing &gt; 3 actors videos and multiple deep faked faces per video and did not want to run in these edge cases in the test set, we decided against the idea.

Using more frames per video was superior to using TTA in our experiments.
We clipped prediction using 0.01 and 0.99.

## Acknowledgments
Thank you to @ratthachat @hmendonca @hmmoghaddam for the incredible experience and Congratulations to @hmmoghaddam on becoming Kaggle expert. 

Thank you to the organizers for hosting this competition.
