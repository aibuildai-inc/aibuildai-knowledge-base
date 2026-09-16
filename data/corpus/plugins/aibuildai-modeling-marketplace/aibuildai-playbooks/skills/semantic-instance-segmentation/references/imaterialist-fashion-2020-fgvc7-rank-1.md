# 1st place solution

Competition: imaterialist-fashion-2020-fgvc7
Rank: #1
Source: https://www.kaggle.com/c/imaterialist-fashion-2020-fgvc7/discussion/154306

# Model



I used a single Mask R-CNN model with the [SpineNet-143](https://arxiv.org/abs/1912.05027) + FPN backbone and added an extra head to classify attributes. The attributes head was trained with the focal loss. For the augmentations, I used one of the [AutoAugment](https://arxiv.org/abs/1906.11172) policies (see details below). No TTAs during the inference.

All the changes were made on top of the [TPU Object Detection and Segmentation Framework](https://github.com/tensorflow/tpu/tree/master/models/official/detection). You can find the **code and weights** for the best model in **[this repo](https://github.com/apls777/kaggle-imaterialist2020-model)**.

Switching from the ResNet-50 to the SpineNet-96 backbone improved my score by on the LB by ~+0.07 (private), ~+0.05 (public). Not sure though how better the SpineNet-143 backbone was as I trained it with a slightly different configuration.

# Data

I split the training data to training and validation datasets in a way that the validation dataset contains at least 10% of images for each class and each attribute. I ended up with 39932 images in the training dataset and 5691 images in the validation dataset.

# Training

* The model was trained on top of pre-trained on the COCO dataset weights.
* It was trained on resolution 1280x1280.
* The attributes head was trained with the focal loss. Switching to the focal loss improved the score by ~+0.012 (private), ~+0.018 (public)
* For the augmentations I used random scaling (0.5 - 2.0) and [v3 policy](https://github.com/tensorflow/tpu/blob/2d9507360e3712715c584e2c21c639b39efd6ad1/models/official/detection/utils/autoaugment_utils.py#L126) from the Google's [AutoAugment](https://arxiv.org/abs/1906.11172) implementation. I modified the code to make it working with masks as it supports only object detection case at the moment.

The model was trained with a batch size 64 for 91.6k steps on a v3-8 TPU for ~69 hours. Big thank you to [TensorFlow Research Cloud](https://www.tensorflow.org/tfrc) for giving me free access to TPUs, it helped a lot!

# Predictions

## Attributes

For the attribute predictions, I used thresholds that maximize F1-score for each individual attribute within a category (so it's 294*46=13524 thresholds, but most of them actually &gt;1 as there are no training examples).

## Best Predictions

I don't think the metric used in this competition was good. Unfortunately, it's not taking into account false-negative predictions at all. That basically means that with just 1 prediction per image the score of 1.0 is still achievable (I first asked about FNs in [this](https://www.kaggle.com/c/imaterialist-fashion-2020-fgvc7/discussion/141891) discussion and later I also contacted the organizer by email to make sure it's not a mistake, but they assured me that the metric is okay.).

So at first I just used one most confident class prediction per image. But high class confidence does not necessarily mean that the segmentation or attributes are good. So next I tried to compute a score for each prediction as an average of class confidence, category mask AP and category attributes F1-score, then I used it to select 1 best prediction per image. It improved my score on the LB by ~+0.041 (private), ~+0.036 (public).

In the end, I implemented [the metric](https://www.kaggle.com/c/imaterialist-fashion-2020-fgvc7/overview/evaluation) used in the competition, computed AP for each model prediction (based on the validation dataset) and trained a regression model to predict APs for the Mask R-CNN predictions. Then, as usual, I just used 1 best prediction per image based on the predicted APs. It improved the score on the LB by ~+0.065 (private), ~+0.058 (public). For the regression model, I ended up using a random forest regressor from the scikit-learn package (100 estimators, max depth 8). It was trained on features like category ID, class confidence, mask area, number of predicted attributes, category mask AP, category attributes F1-score, and so on - 13 features in total.
