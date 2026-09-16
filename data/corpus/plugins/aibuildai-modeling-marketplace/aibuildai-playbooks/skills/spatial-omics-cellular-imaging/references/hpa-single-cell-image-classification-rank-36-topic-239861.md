# HPA 36th Place Solution

Competition: hpa-single-cell-image-classification
Rank: #36
Source: https://www.kaggle.com/c/hpa-single-cell-image-classification/discussion/239861

First of all we would like to thank the host for this really interesting competition, and congrats to all the winners.


# Overview

These are the issues and solutions we had.

**Issue**
① Learning with weak supervised learning
② The nucleus protruding from the image
③ The problem of imbalance in the number of labels

**Solution**
For ③, the problem of imbalance in the number of labels could be reinforced by using external data (downsampling was performed so that each label would have about 10,000 labels). This resulted in an LB score of +0.01. In the past competitions, when the number of labels was unbalanced, using focal loss for loss was a good example of learning, but this time it didn't.

As a measure against weak supervised learning in ①, we used the difference in pixel values. If there were cells with different labels in the image,we assume that the staining intensities are different and that when cropped into a single cell, the average pixel values ​​in the image will vary. Therefore, we contributed to improving the accuracy of the model by using a data set in which 20% of the average value is removed as the threshold value for the average value of the images. In addition, by setting a threshold value, we were able to eliminate a certain number of problems ② in which the nuclei protrude from the image. (Unfortunately, we couldn't compare the thresholds because we didn't have enough time.) As a result, the LB score was +0.03~0.04.


# Models

The models are divided into two types, a single cell model and an image level model.

**Single cell model**
Model：ResNet50+EfficientNet B4
Image size：128x128
Loss：BCEWithLogitsLoss
Augmentation：
Flip
TTA (n=3)
Dataset:
Train：179.2k
Validation：44.7k

**Image level model**
Model：SEResNeXt50 32×4d+EfficientNet B7
Image size：640x640
Loss：
BCEWithLogitsLoss (EfficientNet B7)
FocalLoss (SEResNeXt50 32×4d)
Augmentation：HorizontalFlip(p=0.5)
Dataset:Use only green channel
Train: 17.4k
Validation: 4.4k

That's how I was able to win the silver medal.
And with this medal, I was able to be promoted to Kaggle Master! !
Very glad！

A year before I joined kaggle, I didn't understand Python and machine learning at all, but I feel that by aiming for medals, I've gradually become able to do what I couldn't do. In addition, I think that I was able to train a lot mentally by experiencing a lot of shake-ups and shakedowns XD

Kaggle is the best data science learning platform for me.
Thanks to kaggle and all kagglers.

I will continue to take on challenges
