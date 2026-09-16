# [solution] 10-th place

Competition: siim-acr-pneumothorax-segmentation
Rank: #10
Source: https://www.kaggle.com/c/siim-acr-pneumothorax-segmentation/discussion/107687

The core of the pipeline is still quite similar to my [previous post](https://www.kaggle.com/c/siim-acr-pneumothorax-segmentation/discussion/99440).

One major change is the split of the problem into segmentation and classification. Also after Konstantin joined we have tried more complicated models and a lot of them worked better then ResNet34.

Now, when I see, that most of the team tackled the problem jointly, I start to believe, that having the ensemble of both could significantly increase the score.

**Common:**
Data split: CV5
Optimizer: Adam
Scheduler: Reduce lr on plateau
Augmentations: relatively aggressive: ShiftScaleRotate, Grid- and Elastic-transformation, GaussianNoise

**Classification:**
Models: se\_resnext101 (2 snapshots), senet154
Resolution: 768x768
Loss: BCE
Additional features: TTAx2(hlip), pseudo labeling, gradient accumulation (bs = 100-200)

The key here is the model selection. Instead on focusing on accuracy, I focused on f0.5 metric with fixed threshold of 1.0. The motivation is quite simple, the average dice of segmentation model is around 0.58. It means that correctly guessed negative instance contributes to the score with 1, and correctly guessed positive instance only with 0.58.

**Segmentation:**
Models: Unets: dpn98, se\_resnet101, se\_densenet121 (2 snapshots of each)

The train process consist of 4 stages:
1. Loss: BCE + dice; size: 512x512
2. Loss: BCE + dice; size: 1024x1024
3. Loss: BCE; with soft pseudo labels; size: 1024x1024
4. Loss: symmetric lovasz; size: 1024x1024

Additional features: TTAx6(hlip + rescale), gradient accumulation (bs = 50-100)

**What did not work:**
- Unets with other encoders worked worse (resnet34, resnet50,  dpn107, dpn131, se\_resnext50\_32x4d, se_resnext101\_32x4d, senet154). senet154 worked much better than other models on size 512x512, but completely failed during scaling to 1024x1024.
- Lookahead optimizer didn't improve the optimization process (https://arxiv.org/abs/1907.08610)
- Other losses for classification: FocalLoss, SoftF1Loss. The accuracy with FocalLoss was better then with BCE, but f0.5 metric was worse. 


The code: https://github.com/SgnJp/siim_acr_pneumothorax
