# 6th place solution : noise robust learning [BarelyBears]

Competition: prostate-cancer-grade-assessment
Rank: #6
Source: https://www.kaggle.com/c/prostate-cancer-grade-assessment/discussion/169230

First of all, we would like to thank organizers for such an interesting and realistic problem. 
Implementations are available:
https://github.com/analokmaus/kaggle-panda-challenge-public

# TL;DR
Label noise is the biggest challenge in this competition.
We used **online uncertainty sample mining(OUSM)**  and **mixup** to robustly fit CNN models, and blended 4 models with different settings to stabilize the results. 



# Tile-based multi instance learning model
The first challenge in this comp was how to deal with those extremely large images. Thanks to @iafoss ’s great notebook, we used almost identical model with various backbones and tile sizes. We modified classifier part to ordinal regression ([CORAL loss](https://arxiv.org/abs/1901.07884)).

# Preprocessing and data augmentation
Data augmentation for tile-based CNN model can be applied in two ways: slide level and tile level. Slide level augmentations are applied to whole slide images before tiles are extracted. Since the point is to create slightly different tile sets, we used shift, scale, rotate.
Tile level augmentations aim to improve feature extractor performance, and we used shift, scale, rotate, flip, and random dropout. The idea of random dropout is to randomly fill a tile with mean pixel value and regularize the model.

# Postprocessing
We used 4 times TTA during inference and optimized thresholds to maximize QWK values. 

# Validation strategy
As written in task description, the label quality in train data differs a lot from those in test data. So from the very beginning we assumed this part would be critical in this comp. Roughly speaking, 
train data: noisy and big
public test data: clean but small
private test data: clean but small
So our strategy is **IGNORE CV, CARE ABOUT PUBLIC LB, AND TRUST METHODOLOGY.**
For us, the results were unstable due to small size of test data, but not so ‘lottery’.

# Handling noisy labels
We read tens of papers about handling noisy labels, and implemented some of them such as:

- loss functions (DMI loss, DAC loss, Symmetric loss, **OUSM loss**, etc..)
- training procedure (CleanNet, Iterative Self-training, **mixup**, etc..)

The common ideas among them are that noisy samples should have different features from  correct samples, thus noisy ones should have bigger loss. 
OUSM(Online Uncertainty Sample Mining) is an approach, in which samples with high loss are excluded from each mini-batch. According to [previous research](https://arxiv.org/abs/1901.07759), this method works with skin lesion classification problem where similar kind of label noise exists. In PANDA competition, it gave us stable boost from around 0.87 to 0.90 on public LB. 
Then we trained models with different random seeds, and collected samples which are often judged as noise(with big loss). We excluded 10% of ‘most likely to be noisy’ samples from each label because due to the imbalance in label distribution, grade &gt;= 2 samples are more likely to be judged as noise. This new datasets should be less noisy than the original one, and models trained on this new dataset achieve 0.91 on public LB.
Apart from OUSM, mixup also showed good performance on public LB. This is consistent with [original paper](https://arxiv.org/abs/1710.09412) which reported performance improvement with label corruption.

# Pipeline overview

Our pipeline is simple average of the following models

- 5 fold 224x64Tile-based model, se-resnext50 (OUSM)
- 5 fold 224x64 Tile-based model, se-resnext50 (OUSM with different params)
- 5 fold 224x64 Tile-based model, se-resnext101 (OUSM)
- 5 fold 256x36 Tile-based model, efficientnet-b0 (mixup)
This model scored 0.903 on public LB, and 0.932 on private LB. 
Compared to @haqishen 's model with no denoising, our final model showed +0.018 on public and +0.017 on private.
