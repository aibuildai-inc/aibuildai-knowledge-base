# 8th place solution

Competition: tensorflow-great-barrier-reef
Rank: #8
Source: https://www.kaggle.com/c/tensorflow-great-barrier-reef/discussion/307735

Congrats to all the winners, and thanks to hosts for interesting competition.

# Training

data: annotated images
train: video_id 0, 1 (final submission is all data)
val: video_id 2 (including background images)

# Models

## yolox-s

training size: 2560~3584
augmentation: default parameter
inference size: 2560, 3072, 3584, 4096
tta: WBF conf 0.05
epoch: 12
cv: 0.76

## Cascade-RCNN

backbone: convnext base
training size: 2048~2816
augmentation: RandomGamma, CLAHE, RandomBrightnessContrast, ShiftScaleRotate, Blur,
MotionBlur, GaussNoise
inference size: 2816
epoch: 4
cv: 0.78

# ensemble

WBF conf 0.2
cv: 0.795
public: 0.611
private: 0.726
