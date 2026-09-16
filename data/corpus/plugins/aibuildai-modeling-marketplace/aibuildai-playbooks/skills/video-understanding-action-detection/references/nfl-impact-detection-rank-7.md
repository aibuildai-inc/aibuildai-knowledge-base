# 7-th place solution from AlphaGoal Team [2.5D CenterNet, 3D CNN]

Competition: nfl-impact-detection
Rank: #7
Source: https://www.kaggle.com/c/nfl-impact-detection/discussion/208851

# Intro

Hi there. It was an exciting competition, thanks to the new domain of video analysis. Before I jump into technical details, a brief team intro:

* [Anastasiya Karpovich](https://www.linkedin.com/in/anastasiya-karpovich/) 
* [Tatiana Gabruseva, PhD](https://www.linkedin.com/in/tatigabru/) 
* [Eugen Khvedchenya](https://www.linkedin.com/in/cvtalks/)

For me, it was a great experience to solve this in a team. Since most activities happened during Christmas & NY holidays, we did our best to find the proper balance between life & Kaggling :) The help of my teammates was invaluable, and I'm grateful for being part of this team. @user189546 @blondinka great job, it seems we earned some new GMs & Master badges :)

# Approach 

There were a number of ways to approach this challenge, and here's our way to the gold zone:

We used three stages: helmet detection, impact classification, and post-processing.

The data were split into four folds making sure that both views, Endzone and Sideline, of the same video go to the same fold to avoid data-leak. Each fold has 15 plays (30 videos), which I believe is the same as the size of the test.

## TL;DR:
* 4 models of CenterNet for detection 
* 1 model of 3D CNN for classification
* Postprocessing

## Helmet detection

We experimented with CenterNet and EfficientDet detectors. And our final detection pipeline is what I call 2.5D CenterNet. The model's input was a short clip of 8 consecutive frames, which were passed through the encoder individually (like you would do in siamese networks), then intermediate concatenated and fed through UNet-like decoder to produce output heatmap & impacts map for 8 frames. We found this approach requires much less memory than regular 3D-CenterNet. 

We experimented with different encoders, and DenseNet-based encoders worked the best in terms of Helmet F1 detection and Impact F1 detection scores. EfficientNets worked slightly worse, and ResNet-s performed the worst. On average, F1 score for helmets was around 0.93-0.94, and impact F1 was about 0.43 per fold.

For augmentations, we used albumentations library: https://github.com/albumentations-team/albumentations 

Combined 4 folds with TTA for CenterNet detector gave f1 0.95 with densenet121 backbone and medium-hard augmentations. 2.5D CenterNet ensemble reached the gold zone by itself, but we decided to push forward.

## Impact Classification:

We used several approaches for the classification of the impact. Most of them did not work well enough.

1. Second-level model using 1D CNN on top of the predictions from helmet detection model -- gave around f1 = 0.32 with post-processing (1 fold, no TTA)
2. EfficientDet pre-trained on 1 class for helmet detection and tuned for 2 classes, with medium augmentations and sampling → gave f1 around 0.33  with post-processing (1 fold, no TTA)
3. Mix of the first two approaches -- f1 = 0.34 on LB  (1 fold, no TTA)
4. Siamese networks in videos 
5. 3D CNN classifier on top of the 2D helmets predictions -- this approach gave the best results -- f1 0.57 after post-processing

For 3D CNN classification, we trained only one fold of 3D ResNet50, which, stacked together with predictions of CenterNet, boosted our score by +0.05 or so on the LB. 

## Post Processing

For post-processing we needed to suppress duplicates. We tracked the helmets using IoU, and selected the frame with the maximum impact probability score.
We also tried to use the center of the boxes to track the euclidean distance as a criterion for similar results. 

In the beginning, we removed impacts from the first 30 frames and the last frames, but as the model became more accurate, it did not make a difference.
