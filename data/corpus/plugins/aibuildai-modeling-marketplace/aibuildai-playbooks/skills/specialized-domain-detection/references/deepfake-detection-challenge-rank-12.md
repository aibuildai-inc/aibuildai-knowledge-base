# 12th place in a nutshell

Competition: deepfake-detection-challenge
Rank: #12
Source: https://www.kaggle.com/c/deepfake-detection-challenge/discussion/145822

*Disclaimer: It is my first experience in CV competition, so I apologize if my code or approach insult someone.* 

## Data preparation
- 1st dataset: extracted single random frame from each video
- 2nd dataset: extracted 10 random frames from each video (used for stacking)
- **Blazeface** for face extraction from @humananalog kernel
- output images resized to 256x256
- main augmentations are jpeg compression and downscale with a chance of 0.5
- secondary augmentations (not all were used at the same time): blur, gaussian noise, random brightness, horizontal flip
- classes were balanced by oversampling for single model training and by undersampling for stacked model training

## Models
- **EfficientNet B0-B6** (imagenet)
- **EfficientNet B1-B6** (noisy student)

Efficientnet gave me a huge boost on LB, but more importantly was to choose augmentations and hyperparameters right for this kind of models.

## Training and Validation
I took #40-49 data chunks as validation data and it correlated with LB pretty well (about 10% difference), input size is 256x256 for every model
- single model training: input size depends on model (from 224 to 260), adam 1e-4, StepLR with step 2 and gamma 0.1
- stacked model training: input size 256x256, adam 1e-3

For every model it took about 3-5 epochs to train and start overfitting.

## Stacking &amp; Ensembling
I trained a bunch of decent single models with public LB 0.38-0.4 on the first dataset and then combined different models into a stacking using second dataset. 

Then I took best 5 stacked models from this experiment, their validation score and made frame-wise weighted ensemble

## What didn't work for me
- **MesoNet, RNN**  - underfits on train data
- **Strong augmentations**  (e.g. cutout, huesaturation, 90 rotate)
- **Stacking of stacked models** - seemed like it overfits, but not too much
- **FaceForensics++** - the most I managed to get from this thing is 0.68 public LB

## Links
- **Refactored inference** - https://www.kaggle.com/ims0rry/dfdc-inference-15-lb-private
- **Original inference** - https://www.kaggle.com/ims0rry/inference-demo
- **The whole project on github** - https://github.com/1M50RRY/dfdc-kaggle-solution
