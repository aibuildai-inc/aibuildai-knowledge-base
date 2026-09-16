# 19th Place Solution: Repeated Pseudo Labeling

Competition: hubmap-hacking-the-human-vasculature
Rank: #19
Source: https://www.kaggle.com/c/hubmap-hacking-the-human-vasculature/discussion/428327

First, I would like to thank the competition hosts for organizing this competition and also the participants.

I was disappointed to be kicked out of the gold zone after the shakedown, but the competition was a great learning experience for instance segmentation.

## Overview

My final submission was a single model of Cascade Mask R-CNN (Backbone: ConvNeXt-tiny).

However, the training process had multiple stages. Figure 1 outlines the training and submission pipeline.

.png?generation=1690860001136772&alt=media)
Figure 1: Training and Submission Pipeline

- 1st Training Phase:
Cascade Mask R-CNN was trained using Dataset 1 and 2. Three different backbones (ConvNext-tiny, ResNext-101, Resnet-50) were used for Cascade Mask R-CNN.
Using the three models, we performed pseudo labeling on dataset 3, extracting only labels with a confidence greater than 0.5. We also performed TTA and NMS during pseudo　labeling.
- 2nd Training Phase:
We used Dataset 1 and 2, and the pseudo labeled dataset 3 created in the 1st phase to train the model whose backbone is convnext. In this phase, the model trained in the 1st phase was fine tuned.
- 3rd Training Phase:
Fine tuning the model trained in the 2nd phase using only Dataset 1. This was done to bring the output of the model closer to the annotation quality of dataset 1.
Using the model created here, we performed pseudo labeling on dataset 2 and 3, extracting only labels with a confidence greater than or equal to 0.95. We also performed TTA and NMS during pseudo labeling.
- Final Training Phase:
Dataset 1, and pseudo labeled dataset 2 and 3 created in the 3rd phase were used to train the model whose backbone is convnext. Here, the model trained in the 2nd phase was used for fine tuning.
- Submission:
The model developed in the Final Training Phase was used to infer the test dataset; it is a single model of Cascade Mask R-CNN (Backbone: ConvNeXt-tiny), but we have performed TTA and NMS.

## My Approach

I proceeded to train the model, focusing on the quality of the annotations in each dataset. The following is a quote from the description of the datasets in this competition.

> The competition data comprises tiles extracted from five Whole Slide Images (WSI) split into two datasets. Tiles from Dataset 1 have annotations that have been expert reviewed. Dataset 2 comprises the remaining tiles from these same WSIs and contain sparse annotations that have not been expert reviewed.
> 

> All of the test set tiles are from Dataset 1.
> 

> Two of the WSIs make up the training set, two WSIs make up the public test set, and one WSI makes up the private test set.
> 

> The training data includes Dataset 2 tiles from the *public* test WSI, but *not* from the *private* test WSI.
> 

> We also include, as Dataset 3, tiles extracted from an additional nine WSIs. These tiles have not been annotated. You may wish to apply semi- or self-supervised learning techniques on this data to support your predictions.
>

This means,

- Dataset 1 → accurate annotations and the same annotation quality as the test dataset
- Dataset 2 → not so accurate annotations
- Dataset 3 → no annotations

Therefore, I decided to trust Dataset 1 and tried to improve the quality of the annotations output by the model to be closer to Dataset 1. I also tried to make good use of the remaining Dataset 2 and 3. Based on this policy, I proceeded with my experiments.

- Model
    - convnext-tiny mainly used.
    - resnet, resnext were also used, but finally not used because they did not give good results when used with pseudo label.
        - resnet and resnext were used for the first pseudo labeling.
    - mmdetection 3.x was used.
- Annotations Type
    - Only blood_vessel was used.
    - unsure, glomerulus not used at all.
- Data Augmentation
    - The images were randomly augmented to multiple scales during training as shown below.
        - scales=[(640, 640), (768, 768), (896, 896), (1024, 1024), (1152, 1152), (1280, 1280), (1408, 1408), (1536, 1536)]
- Test Time Augmentation (TTA)
    - The ensemble of output results from multiple scales was used.
        - scales=[(1024, 1024), (1280, 1280), (1536, 1536)]
- Post-Processing
    - Non-Maximum Suppression (NMS) was used during TTA.
    - No dilation.
- Pseudo Labeling
    - Pseudo labeling was performed in two parts as shown in Figure 1. TTA was performed for three different image sizes, scales=[(1024, 1024), (1280, 1280), (1536, 1536)], and NMS was used for the ensemble.
- Cross Validation (CV) Strategy
    - CV was set to be similar to private LB.
        - Source WSI == 1 & dataset == 1 → valid
        - Source WSI == 1 & dataset != 1 → It was not used for cv.
        - Others → train
    - At the time of submission, all data was used for training.
    - CV was not always correlated with Public LB. However, it became more correlated as the experiment progressed to the latter part of the experiment.

## What Worked

- Pseudo Labeling
- Trust the quality of the annotations in Dataset 1
- TTA
- NMS
- Scale Up Images

## What Didn’t Worked

- Models with large parameter sizes
    - mask2former, convnext-small
    - Maybe because of lack of parameter tuning…?
- Use unsure and glomerulus for training
- Dilation
    - LB was effective when raw dataset 2 was used for training data
    - LB was decreased when using only dataset 1 as training data
- Flip Data Augmentation
- Weighted Boxes Fusion (WBF) (my implementation may have been suspect…)
- Smooth inference around edges (my implementation may have been suspect…)
    - Reference: https://www.kaggle.com/competitions/hubmap-kidney-segmentation/discussion/238013
- Ensemble with other models

## Code
- GitHub (My training code. I'm sorry but it's not well maintained) -> https://github.com/moritake04/hubmap-2023
- Submitted notebook -> https://www.kaggle.com/code/moritake04/private19th-final-sub
