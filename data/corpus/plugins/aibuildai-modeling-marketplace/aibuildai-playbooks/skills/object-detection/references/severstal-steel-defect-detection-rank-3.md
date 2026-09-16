# 3rd Place Solution

Competition: severstal-steel-defect-detection
Rank: #3
Source: https://www.kaggle.com/c/severstal-steel-defect-detection/discussion/117377

Sorry to share so late . First of all thanks the organizer Severstal and Kaggle ​ for a very interesting and challenging competition. I hope all the  solutions could bring some help and ideas for steel defect detection. Thank everyone that share the ideas in this competition. Without your help, I can't get this gold. I have to thank @hengck23 ! After trying a lot of model pipeline, I found his solution is the most  concise one and my final solution is based on that. 

## Score 
Winning submission:
| Public LB        | Private LB  | 
| ------------- |:-------------:|
| 0.91863 |  0.90765 | 

Best submission 
| Public LB        | Private LB  | 
| ------------- |:-------------:|
| 0.91824 |  0.90934 | 

## Summary
**Basic Model:​** Unet, Feature Pyramid Network (FPN)
**Encoder:**​  efficientnet-b3, efficientnet-b4, efficientnet-b5, se-resnext50
**Loss:​** Focal Loss
**Optimizer:​** Adam, init lr = 0.0005
**Learning Rate Scheduler:​** ReduceLROnPlateau (factor=0.5, patience=3,cooldown=3, min_lr=1e-8)
**Image Size:**​ 256x800 for training, 256x1600 for inference
**Image Augmentation:**  horizontal flip, vertical flip
**Sampler:** Weighted Sampler 
**Ensemble Model**:
I simply average 9 model output probability to achieve the final mask probability without TTA
1. FPN + efficientnet-b5 + concatenation of feature maps
2. FPN + efficientnet-b4
3. Unet + efficientnet-b4 , add pseudo labeling data in training data
4. Unet + efficientnet-b4, training with heavy image augmentation
5. Unet + efficientnet-b4 +SCSE layer
6. Unet + efficientnet-b4 +SCSE layer, add pseudo labeling data in training data
7. Unet + efficientnet-b4 + Mish layer
8. Unet + efficientnet-b3
9. Unet + se-resnext50

**Threshold**
Label Thresholds: 0.7, 0.7, 0.6, 0.6
Pixel Thresholds: 0.4, 0.4, 0.4, 0.4

## Model Pipeline


You can check this  [Link](https://www.kaggle.com/c/severstal-steel-defect-detection/discussion/111457#latest-654845) for more detail and code, thank to HengCherKeng again !

## The key point I think that help me win a gold 
- use code pipeline https://github.com/PavelOstyakov/pipeline
- make sure the diversity of models in ensemble
- using one-stage model pipeline prevent me from training classification model individually and tuning too much hyper-parameters. Simple is best.
