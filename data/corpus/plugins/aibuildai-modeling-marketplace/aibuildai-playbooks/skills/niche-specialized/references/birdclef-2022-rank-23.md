# 23th place solution

Competition: birdclef-2022
Rank: #23
Source: https://www.kaggle.com/c/birdclef-2022/discussion/327046

# TL;DR

My solution features the following.

* Local validation:
  - train with all un-scored 131 species + 66% of scored 21 species, validate with 33% of scored 21 species
  - StratifiedGroupKFold (Group by `author`)
* Ensemble of PANNs + PaSSTs (GeM pooling (p=3))
* Thresholding per group (fixed quantile)

## Local Validation Strategy

* All 131 un-scored species are used for training
* 21 scored species were split into 3-folds with `StratifiedGroupKFold`

To see the effect of domain shift, I made sure that the same Author did not appear in both training and evaluation data [1].

<a href="https://ibb.co/VQrSQdP">[Screen-Shot-2022-05-25-at-19-36-58]</a>

## Model

The model consists of an ensemble of PANNs and PaSSTs.

### PANNs

The code for PANNs was adapted from the 2020 6th rank solution [2].
The changes are as follows.

* time window is set to 20 seconds
* mixup + cutmix (adapted from a public notebook [3])
* backbone: ResNet-34
* use FocalLoss
* use AdamW
* train 40-100epoch

### PaSST

The source code was adapted from the official implementation [4]. The changes are as follows.

* audio-based augmentation such as Gauss noise
* Knowledge distillation using pseudo-labels from learned PANNs (ResNet-34)
* Use of FocalLoss
* Using AdamW
* train 40 epochs

### About Knowledge Distillation

The loss function is the average of the loss with the pseudo-label as the label and the loss calculated with the original correct label.

`loss(pred, y, y_pseudo_label) = 0.5 * (loss(pred, y) + loss(pred, y_pseudo_label))`

## Ensemble Strategy

Predictions from a total of 8 models (PANNs x4 + PaSST x4) were aggregated by GeM pooling (p=3).

## Thresholding Strategy

Following the 2021 second-order solution[5], I fixed the quantile of the predictions and set the threshold [6].
Where I changed is that I divide the scored species into the following four groups, and set threshold per each group.

* top5: ['skylar', 'houfin', 'jabwar', 'warwhe1', 'yefcan']
* mid_top5: ['apapan', 'iiwi', 'omao', 'hawama', 'hawcre']
* mid_low5: ['barpet', 'akiapo', 'elepai', 'aniani', 'hawgoo']
* low6: ['ercfra', 'hawpet1', 'puaioh', 'hawhaw', 'crehon', 'maupar'])

The thresholds for each group of the final submitted model are as follows. These were optimized for Public LB.

top5, mid_top5, mid_low5, low6 = [0.100, 0.550, 0.350, 0.334]

# Evaluation Results

The evaluation results for the top submissions, including the final submission, are shared below.

Updated: the result of late submission (sub10) shows ensemble without PaSST is slightly better than that includes PaSST (0.7626->0.7645). So, the PaSST have no positive effect on my experiments.

<a href="https://ibb.co/vX1ymnS">[Screen-Shot-2022-05-27-at-22-37-59]</a>

# What didn't worked

* soft balanced accuracy loss
* binary classifier

## References.

* [1] https://www.kaggle.com/code/tatamikenn/birdclef22-meta-sub-clip-60sec-group-by-author/notebook
* [2] https://www.kaggle.com/competitions/birdsong-recognition/discussion/183204
* [3] https://www.kaggle.com/code/kaerunantoka/birdclef2022-use-2nd-label-f0
* [4] https://github.com/kkoutini/PaSST
* [5] https://www.kaggle.com/competitions/birdclef-2021/discussion/243463
* [6] https://www.kaggle.com/code/tatamikenn/birdclef22-sub7-1-3-6-8-9-10-passtx4-panns-x4/notebook
