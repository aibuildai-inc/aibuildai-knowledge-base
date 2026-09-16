# 3rd place solution with code.

Competition: human-protein-atlas-image-classification
Rank: #3
Source: https://www.kaggle.com/c/human-protein-atlas-image-classification/discussion/77320

## UPDATE: code available on github
https://github.com/pudae/kaggle-hpa

---

Congrats to all the winners and thanks to all kagglers who posted great discussions. It was very helpful to me.

Thanks to Kaggle and HPA team for an interesting competition.

Here is overview of my solution.

## Dataset Preparation
Like almost the other competitors, I also used official + [external](http://www.proteinatlas.org) data. 
(Thanks to [TomomiMoriyama](https://www.kaggle.com/tomomimoriyama) and [David Silva](https://www.kaggle.com/dr1t10))

I splited dataset as following:

* 1/10 holdout set for ensemble.
* building 5 folds cross validation sets using rest of 9/10.
* using phash and ahash to prevent duplicate images in difference splits. If the labels are not matched between official and external, I used official one. (Thanks to [Tilii](https://www.kaggle.com/tilii7))

## Input Preprocessing
I found that the distributions of image mean and stddev are very difference between official and external. So, I used mean and stddev of individual images for input normalization.

## Augmentation
I searched suitable data augmentation as following [AutoAugment](https://arxiv.org/pdf/1805.09501.pdf). For simplicity, I used random search instead of RL.

## Loss
Focal loss with gamma 2.

## Training
* Adam optimizer, learning rate 0.0005.
* no learning rate scheduling.
* For the large model with 1024x1024 images, I used gradient accumulation so that weights are updated every 32 examples.
* Early stopping
  * If I choose checkpoints that record best macro F1 score for the validation set, LB scores are poor.
  * After analyzing F1 scores of each classes, I found that while macro F1 score is increasing, F1 scores of high-proportion classes (like 0, 1) are decreasing. Because of relying on rare class score is risky, I decided to stop training when F1 score of 0 class is decreasing.

## Inference
* Averaging the weights of last 10 checkpoints.
* 8 test time augmentation
* weighted averaging ensemble

## Thresholds
Because of rare classes, macro F1 score is very sensitive to thresholds. I tested various method for finding good threshold, but almost tries are failed.

My final method is following:

* For each classes, I choose the thresholds that make the proportion of positive predictions in validation set are closed to the proportion of positive examples. (Thanks to [lafoss](https://www.kaggle.com/iafoss) for the LB probing)

## Models
**512x512**

* resnet34: 5 fold ensemble with TTA: Public LB 0.574 / Private LB 0.500

**1024x1024**

* inceptionv3: single fold with TTA: Public LB 0.583 / Private LB 0.549
* se_resnext50: single fold with TTA: Public LB 0.601 / Private LB 0.531
* From 1024x1024, the mean and stddev of individual images are used.
* In case of 1024x1024 input, using global average pooling is not good performance in my case.. maybe... So, I modified last layer following:
  * remove global average pooling.
  * compute MxM logits using 1x1 convolution.
  * compute weight maps using 1x1 convolution followed by softmax.
  * using weight maps, compute weighted averaged logits.
* Final submission is ensemble of above three predictions.
* Additional models are trained, but the ensemble results were not good.

Because I failed to make stable CV, I can't be sure that methods described above were effective. Finding good methods without stable CV was painful process. So, I hope to learn from the [bestfitting](https://www.kaggle.com/bestfitting)'s solution that produce stable results always.
