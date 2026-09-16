# 10th place solution

Competition: understanding_cloud_organization
Rank: #10
Source: https://www.kaggle.com/c/understanding_cloud_organization/discussion/118262

Congratulations to all winners!
Here is my solution (Public 0.67376, Private 0.66765).

## Pre processing:
- resize image size to (320, 512)
- exclude bad images (removed 21 images)

## Augmentations:
I used [albumentations](https://github.com/albu/albumentations).
- HorizontalFlip, VerticalFlip
- ShiftScaleRotate, GridDistortion
- Blur, MedianBlur, GaussianBlur
- CLAHE, RandomBrightnessContrast, HueSaturationValue, IAASharpen
   
## Model:
I used [segmentation_models.pytorch](https://github.com/qubvel/segmentation_models.pytorch),  [pretrained-models.pytorch](https://github.com/cadene/pretrained-models.pytorch),  [EfficientNet-PyTorch](https://github.com/lukemelas/EfficientNet-PyTorch).

#### Model-1:
- densenet169 Unet with classification
- image size : 320x480

#### Model-2:
- efficientnet-b4 FPN with classification
- image size : 320x480
   
## Optimizer:
- [RAdam](https://github.com/LiyuanLucasLiu/RAdam)
   
## Loss:
#### Pre training:
- segmentation: BCE + Dice
- classification: FocalLoss

####  Main training:
- (classification)( FocalLoss * 0.5 + BCEWithLogits * 0.5 ) * 0.05 + (segmentation)( BCE + Dice ) * 0.95
   
## Ensemble:
- simple average of the 2 models(x 4 = total 8 models)
   
## Post processing:
- TTA : None, h-flip, v-flip, h-flip and v-flip
- Threshold : I use [optuna](https://github.com/optuna/optuna) to find the optimal value from the cv score. 
   
## GPU:
- RTX2080Ti x 1
