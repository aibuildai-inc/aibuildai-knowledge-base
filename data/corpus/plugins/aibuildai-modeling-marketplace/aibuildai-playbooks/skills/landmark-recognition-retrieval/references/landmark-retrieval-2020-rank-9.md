# 9th place solution overview

Competition: landmark-retrieval-2020
Rank: #9
Source: https://www.kaggle.com/c/landmark-retrieval-2020/discussion/175472

We thank all organizers for this very exciting competition.
Congratulations to all who finished the competition and to the winners.

We trained models using only GLDv2clean dataset on PyTorch and converted them to TensorFlow’s saved_model.
Then, we validated them using GLR2019 public and private datasets.

## Final submission

ResNeSt50 + ResNet101 + ResNet152 + SEResNeXt101

|model                    | output dim  | input size | GLR2019 mAP@100 | Public LB | Private LB |
|:------------------------|:------------|:-----------|:--------|:----------|:-----------|
|ResNeSt50                |512          |416         | 0.3114  | 0.335     | 0.292      |
|ResNet101                |512          |608         | 0.3243  | 0.346     | 0.303      |
|ResNet152                |512          |480         | 0.3180  | 0.336     | 0.288      |
|SEResNeXt101             |512          |480         | 0.3209  | 0.337     | 0.296      |
|Ensemble of 4 models     |2048 (concat)|-           | 0.3396  | 0.361     | 0.317      |


## Model details
- Backbones: Ensemble of ResNeSt50, ResNet101, ResNet152 and SEResNeXt101
- Pooling: GeM (p=3)
- Head: FC->BN->L2 (the same as the last year’s first place team)
- Loss: CosFace with Label Smoothing (ArcFace was also good, but CosFace was better)
- Data Augmentation: RandomResizedCrop, Rotation, RandomGrayScale, ColorJitter, GaussianNoise, Normalize, and GridMask
- LR: Cosine Annealing LR with warmup, training for 30 epochs
- Input image size in training: 352

## What we tried and worked
- Automatic mixed precision training
- Replace GeM p=3 with p=4 in testing
- Increase input image size last few epochs of training with freezed BN
- Large and multiple input image sizes in testing: 416, 480 and 608
- MVArcFace

## What did not work
- PCA whitening
- Maintaining the aspect ratio of input images in testing (perhaps because our models were trained on square images)
- Combination of arcface loss and pairwise (e.g., triplet) loss
  We have tried arcface loss with pairwise loss (specially multi-similarity loss). However, the single arcface loss was better than multiple losses.
- EfficientNet
- Circle loss
- Removing noisy classes
  We have tried to remove the worst 3 noisy classes which have high variance based on arcface class weight, but it did not work.

## What we have not tried
- Training with GLDv1 dataset
- Training without changing the aspect ratio of images
