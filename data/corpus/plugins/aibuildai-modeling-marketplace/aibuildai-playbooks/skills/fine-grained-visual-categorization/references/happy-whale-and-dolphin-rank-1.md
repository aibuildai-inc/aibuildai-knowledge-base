# 1st Place Solution

Competition: happy-whale-and-dolphin
Rank: #1
Source: https://www.kaggle.com/c/happy-whale-and-dolphin/discussion/320192

This was my first participation in a Kaggle competition and I was so fortunate to win 1st place!!!!
I appreciate my team member @charmq!

We really enjoyed the competition and worked hard literally until the last minutes. We would like to deeply appreciate the Kaggle staff for organizing this great competition and other teams for competing with us.

## Overview
Our solution is based on sub-center ArcFace with Dynamic margins, which was shown to be effective in [Google Landmark Recognition 2020 the 3rd place solution](https://www.kaggle.com/competitions/landmark-recognition-2020/discussion/187757)  by @boliu0 and [the first place solution of Google Landmark Recognition 2021](https://www.kaggle.com/c/landmark-recognition-2021/discussion/277098) by @christofhenkel.

Basically, our solution was an ensemble of two pipelines implemented by @charmq and me. Since we shared knowledge with each other during the competition, our pipelines share much in common. Below, I will mainly explain my pipeline, which was slightly better in CV score. 

## Key Points
- Tuning of dynamic margin hyperparameters with [Optuna](https://optuna.org/)
- Larger learning rate for ArcFace head
- Bounding box mixing augmentation
- Ensemble of knn and logit
- Two-round pseudo labeling
- Ensemble of many models

## Dataset
We used several types of bounding boxes to crop images. We thank @jpbremer and @phalanx a lot for providing valuable datasets. We also trained our own yolov5 model using fullbody annotations, which we refer to as fullbody_charm.

For train data, we randomly mixed several bboxes with the ratio of fullbody:fullbody_charm:backfin:detic:none=0.60:0.15:0.15:0.05:0.05. Especially, combining backfin bbox to train data significantly improved the performance possibly because it enhances the robustness to images that only contain backfins. Adding non-cropped images by a small ratio also worked as a regularization. For test data, we took the mean of predictions between fullbody and fullbody_charm.

We used the cropped images resized to a fixed size. We mainly used the image size of (1024, 1024). Some models were trained with the image size of (1200, 1200) and (1440, 1440) for ensembling.

## Backbone & Neck
We trained several different imagenet-pretrained backbones for ensembling (efficientnet_b5, efficientnet_b6, efficientnet_b7, efficientnetv2_m, efficientnetv2_l, etc). The best performance in a single model was achieved by efficientnet_b7.

Using GeM pooling (p=3) instead of GAP enhanced the performance.

The normalization layer before the ArcFace head was important. Batchnorm was slightly better than Layernorm in our experiments.

In addition to the final feature map of the backbone, we used the second final feature map to capture more local information. We simply concatenated those two GeM-pooled feature maps and passed them to head.

## Head
For handling imbalanced classes, we adopted ArcFace with dynamic margins. Since it seemed sensitive to hyperparameters, we tuned them on images of (256, 256) and efficientnet_b0 using Optuna. It seemed the acquired hyperparameters also worked well on large images and architectures.

In the last competition, it was reported that handling flipped images as different classes significantly enhanced the performance. In this competition, we did not think that this technique works well because some images are taken from different angles. To handle this issue, we adapted the sub-center ArcFace of k=2 with the usual flip data augmentation.

We also added a second head for classifying species. Sub-center ArcFace with dynamic margins worked better than simple Linear head.

## Training
While we mainly checked a single-fold validation score locally, we trained our models by using whole train data for submission.
Setting the learning rate of the head 10 times bigger than the learning rate of the backbone significantly improved the performance.
Optimal training settings of us differed possibly due to slight differences in our pipelines. While I trained the models for 30 epochs by AdamW optimizer of lr_backbone=1.6e-3 with warmup cosine annealing scheduler, charmq trained the models for 20 epochs by Adam of lr_backbone=1e-4 with cosine annealing scheduler. Most of the models were trained with the batch size of 16-32 on 2-8x NVIDIA Tesla V100 (32GB).

In addition to the bounding box mix augmentation described above, we adopted many data augmentations because the models were expressive enough to reach almost 100% of training accuracy. Below are the list of data augmentations we used by Albumentations implementation:
```
A.Affine(rotate=(-15, 15), translate_percent=(0.0, 0.25), shear=(-3, 3), p=0.5),
A.RandomResizedCrop(image_size[0], image_size[1], scale=(0.9, 1.0), ratio=(0.75, 1.3333333333)),
A.ToGray(p=0.1),
A.GaussianBlur(blur_limit=(3, 7), p=0.05),
A.GaussNoise(p=0.05),
A.RandomGridShuffle(grid=(2, 2), p=0.3),
A.Posterize(p=0.2),
A.RandomBrightnessContrast(p=0.5),
A.Cutout(p=0.05),
A.RandomSnow(p=0.1),
A.RandomRain(p=0.05),
A.HorizontalFlip(p=0.5),
```

## Postprocess
We combined the following two metrics.
knn: Using feature vectors, we calculated the largest cosine similarity for each class in training data. This can be done for each model and is easier to ensemble many models than feature concatenation. In implementation, we looked at only the top 500 training data for each test individual by `sklearn.neighbors.NearestNeighbors`. As a test-time augmentation, we took the mean of original test images and flipped test images searched over both original train images and flipped train images.
logit: We used the simple output of the model without margins.
(We calculated the mean of predictions for two bounding boxes as explained above.) While knn worked much better in CV but only slightly better in the public leaderboard than logit. This is probably caused by highly imbalanced data and the distribution differences between train and test (knn is more likely to output classes with more train data). To mitigate this, we mixed the prediction of knn and logit with knn_ratio=0.5. After pseudo labeling, we increased the knn_ratio to 0.8.

We labeled individuals whose ensembled predictions are lower than a certain threshold as “new_individual”. Through several submission trials, we decided to set the threshold so that the ratio of “new_individual” as the first prediction is 0.165.

## Other
Pseudo labeling enhanced the public LB score a lot in this competition probably because of extremely imbalanced data. We stuck to improving CV scores until the very final stage of the competition. On the day before the deadline, we got a big boost in the leaderboard score (0.88589/0.85959 -> 0.89343/0.87062) by a pseudo-label submission. The second round of pseudo labeling on the final day also improved the score (0.89680/0.87579). Additional rounds of pseudo labeling might have further improved performance, but unfortunately, we did not have time to do that.

As a final submission, we ensembled around 50 models including the ones trained without pseudo labels and with the first-round pseudo labels. After the competition, we confirmed that the ensemble of only 2 models (the best one from each of us) scored 0.89385/0.87336, which could still win first place!

## What did not work
- input 4-channel images with segmentation mask (1st place solution of the last competition)
- input 6-channel images combining 2 types of images cropped by fullbody and backfin bboxes
- input rectangle images such as (512, 1024)
- maintain aspect ratio of images by bounding box expansion instead of resizing
- AdaCos, triplet loss
- Focal loss
- training p of GeM pooling
- ConvNeXt
- Swin Transformer (384 was too small)
- dolg
- using pseudo labels for knn
- cutmix

## Acknowledgement
We deeply acknowledge great OSS such as PyTorch, PyTorch Lightning, PyTorch Image Models, Albumentations, etc. We would also like to appreciate Preferred Networks, Inc for allowing us to use computational resources.

[Update 2022/05/24]
I added explanations on some minor points.
code (knshnb): https://github.com/knshnb/kaggle-happywhale-1st-place
code (charmq): https://github.com/tyamaguchi17/kaggle-happywhale-1st-place-solution-charmq
