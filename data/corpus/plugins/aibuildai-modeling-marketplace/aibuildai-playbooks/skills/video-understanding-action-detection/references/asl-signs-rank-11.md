# 11th place solution with code

Competition: asl-signs
Rank: #11
Source: https://www.kaggle.com/c/asl-signs/discussion/406657

Thank you to the organizer and Kaggle for hosting this interesting challenge.
Especially I enjoyed this strict inference time restriction. It keeps model size reasonable and requires us for some practical technique.

## TL;DR
- Ensemble 5 transformer models
- Strong augmentation
- Manual model conversion from pytroch to tensorflow
- Code is available here -> https://github.com/bamps53/kaggle-asl-11th-place-solution

## Overview
I started from @hengck23 ‘s [great discussion](https://www.kaggle.com/competitions/asl-signs/discussion/391265) and [notebook](https://www.kaggle.com/code/hengck23/lb-0-67-one-pytorch-transformer-solution). Thanks for sharing a lot of useful tricks as always!


The changes I made are following;
- Change model architecture to CLIP transformer in HuggingFace
- Decrease parameter size to maximize latency within the range of same accuracy
- Some strong augmentations
    -  Horizontal flip(p=0.5)
    -  Random 3d rotation(p=1, -45~45)
    -  Random scale(p=1, 0.5~1.5)
    -  Random shift(p=1, 0.7~1.3)
    -  Random mask frames(p=1, mask_ratio=0.5)
    -  Random resize (p=1, 0.5~1.5)
- Add motion features
    - current - prev
    - next - current
    - Velocity  
- Longer epoch, 250 for 5 fold and 300 for all data

For the details, please refer to the code.(planning to upload)

## Model conversion
I’m too lazy to implement augmentations in tensorflow dataset, so I keep using pytorch training pipeline. But I’ve realized that inference time of the model converted by onnx_tf is way slower than bare tensorflow models. Then I’ve tried some model conversion framework like nobuco, but there were too many errors for some reason. Finally I’ve decided to write the model architecture both in pytorch and tensorflow, then manually port the weight. Thankfully HuggingFace has both pytorch and tensorflow CLIP implementation, this work is easier than I thought. This significantly speeds up inference time and I can put more models when ensembling.

## Ensemble
I’ve tried to diversify the models as much as possible within the same accuracy range.

| seed | layers | dim | act  | max_len | features            |
|------|--------|-----|------|---------|---------------------|
|    0 |      2 | 384 | relu |      64 | lip/hand            |
|    1 |      3 | 256 | relu |      48 | lip/hand/eye/motion |
|    2 |      2 | 384 | geru |      64 | lip/hand            |
|    3 |      2 | 384 | geru |      64 | lip/hand/eye/motion |
|    4 |      3 | 256 | geru |      48 | lip/hand/eye/motion |


The scores are as follows;

|                   | public | private |
|-------------------|--------|---------|
| Single best model |  0.786 |   0.865 |
| Ensemble 5 models |  0.794 |    0.87 |


## What didn’t work;
- Normalization matters a lot, so I’ve tried a lot of variants but couldn’t get a better result than simple video mean/std normalization.
- More landmarks(like arm, ear, nose or pose)
- Conv1d
- Knowledge distillation
- Model soup
- Bigger model
- Pretrained model (ex. CLIP pretrained model or pretrain with NTU dataset)
- And so on..

## Code
https://github.com/bamps53/kaggle-asl-11th-place-solution
