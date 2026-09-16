# 6th Place Solution (41st in the Public LB)

Competition: airbus-ship-detection
Rank: #6
Source: https://www.kaggle.com/c/airbus-ship-detection/discussion/71782

First of all, I'd like to thank my teammates [ZFTurbo][1] and [Nick Sergievskiy][2] for the nice teamwork and their great effort!

We've entered the competition and merged into the team pretty late. Thus, we haven't had enough time to make lots of experiments. Another problem was the validation. While our local validation score was growing, Public LB score remained the same, at about 0.738, throughout the last week. The best reward for this struggle was a move from the 41st place in the Public LB to the 6th place in the Private LB.

There have been already shared lots of great methods and ideas from the top teams. So, here is our brief solution outline.

## Local Validation
We created 5 folds validation without a leak. As already mentioned, there were inconsistencies between Local and Public LB score movement. However, we tried to trust only the local validation.

## Models
 1. Classification (empty vs non-empty images). InceptionResNetV2, trained on 299x299 by [ZFTurbo][1]
 2. Semantic segmentation: ResNet34 + U-Net by me. Trained on 256x256 random crops, prediction on a full-size 768x768
 3. Semantic segmentation: ResNet152 + U-Net by [ZFTurbo][1]. Trained on 224x224 random crops, prediction with a sliding window
 4. Instance segmentation: ResNet18/ResNet50 + Mask R-CNN trained on 1000x1000 by [Nick Sergievskiy][2]

## Sampling
We used different percentages of non-empty / empty images in the batch. It was 50/50 for ResNet34 U-net, and even 90/10 for ResNet152 U-net. So, it generated lots of False Positive ships, and the role of the Classifier was pretty crucial.

## Performance
 1. Ensemble of 4 models + TTA for ResNet34 + U-Net. Private LB: 0.846 -&gt; 0.848 with Classification (deleting masks for confident empty images)
 2. Ensemble of 3 models + TTA for ResNet152 + U-Net. Private LB: 0.775 -&gt; 0.848 with Classification
 3. Ensemble of 3 models + TTA for Mask R-CNN. Private LB: 0.843 -&gt; 0.850 with Classification

## Final Ensemble
As a final ensemble we used:

 1. Geometric mean of 7 U-Net models including one model with pseudolabels and one 2nd level model on OOF predictions. Denote predictions of this ensemble as **unet\_mask**
 2. Ensemble of 3 Mask R-CNN models

For the Mask R-CNN ensemble [Nick Sergievskiy][2] chose two thresholds: thr\_high and thr\_mid. They gave the most confident predictions (**rcnn\_mask\_high**) and just confident predictions (**rcnn\_mask\_mid**). Further, **rcnn\_mask\_high** had the highest priority and replaced **unet\_mask** objects; **rcnn\_mask\_mid** were added only if there was an intersection with **unet\_mask** objects.

[1]: https://www.kaggle.com/zfturbo
[2]: https://www.kaggle.com/nicksergievskiy
