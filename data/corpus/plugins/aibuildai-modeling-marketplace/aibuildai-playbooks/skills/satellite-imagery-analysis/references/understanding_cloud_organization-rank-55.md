# Easy silver in last days [55th]

Competition: understanding_cloud_organization
Rank: #55
Source: https://www.kaggle.com/c/understanding_cloud_organization/discussion/118019

## Easy silver in last days
I have adopted my pipeline from Severstal Defect Detection and was able to get silver medal in last two days with just 6 submissions, here is a short description of 55th place solution.

2 step pileline
 1) Multi-task network (classification + segmentation) as classifier to remove empty masks
 2) Binary segmentation for each class
 
### 1st step.
I have trained 5-fold `FPN(resnet34) + aux classfication output` on  480x640 images using `Flip`, `RandomBrightness` as augmentations. Model trained just 6-7 epochs and than starts to overfit, I do nothing with that, just save top 5 checkpoints according to metric.

Loss (segmentation head): bce+dice
Loss (classification head): bce
Optimizer: AdamW
Postprocessing: remove masks less than 10000 pixels
Thresholds: [0.6, 0.6, 0.6, 0.6]


### 2nd step.
For each class trained `2 x Unet(se_resnext50_34x4d)` only on images with masks of that class!
with same optimizer, image size and augmentations.

Loss: bce+dice
Thresholds: [0.4, 0.4, 0.4, 0.4]

### Ensemble
For all models made checkpoints weights! averaging (+0.005-0.01 on validation).
Models over each stage have been just averaged with Flip TTA.

### Useful links
 - Segmentation Models: https://github.com/qubvel/segmentation_models.pytorch
 - Test Time Augmentation for PyTorch: https://github.com/qubvel/ttach

**And congratulations to winners!**
