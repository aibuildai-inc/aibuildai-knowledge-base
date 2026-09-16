# 13th place solution

Competition: understanding_cloud_organization
Rank: #13
Source: https://www.kaggle.com/c/understanding_cloud_organization/discussion/118072

Thanks Kaggle and Max Planck Institute for this interesting competition and congrats to all the winners!  Here is a brief summary of my solution (Public 0.67698, Private 0.66713).


##Preprocessing
- exclude bad images (removed 13 images)
- resize image size to (320, 512)


##Augmentations (by Albumentations)
- gamma (limit=(50,100), p=0.5)
- brightness (limit=0.2, p=0.5)
- shift (limit=0.2, border_mode=0, p=0.5)
- rotation (limit=30deg, border_mode=0, p=0.5)
- horizontal flip (p=0.5)
- vertical flip (p=0.5)


##Validation
- StratifiedKFold for the number of empty masks


##Model (ensemble of 7 models x 5folds)
1. UNet-ResNet34 + CBAM + Hypercolumns
2. same as 1. but with other seed
3. UNet-ResNet18 + CBAM + Hypercolumns
4. UNet-InceptionResNetV2 + CBAM+ Hypercolumns
5. UNet-SeResNext50 + CBAM + Hypercolumns
6. UNet-ResNet34 + CBAM + FPA
7. UNet-ResNet18 + CBAM + FPA
I used the weights of best validation score epochs.


##Loss
- BCE + LovaszHinge
- on top oh that I used deep supervision with BCE+LovaszHinge loss (for only non-empty masks) multiplied by 0.1


##Optimizer &amp; Scheduler
- Adam &amp; CosineAnnealingWarmRestart (20epoch cycle)
- learning rate : 1e-4 to 1e-6


##Ensemble 
- simple average of the 7 models (x 5folds = total 35 models)


##Postprocessing
- TTA : None + h-flip + v-flip + h- and v- flip
- pixel threshold = 0.45
- small mask threshold = 18000
Both determined by the 5foldCV for model 1.


##Final submission
- I checked only Public LB score for ensembles. So I needed some criteria to choose the final submission. I decided to choose two submissions which were good in Public LB and stable against the small mask threshold, although these were not my best Public LB submission. Luckily I survived the shake up and got a gold medal.
