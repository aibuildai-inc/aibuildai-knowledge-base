# 2nd place solution

Competition: siim-acr-pneumothorax-segmentation
Rank: #2
Source: https://www.kaggle.com/c/siim-acr-pneumothorax-segmentation/discussion/108009

Congrats to all winners. Many thanks to Kaggle and the organizers for holding this amazing competition. And special thanks to my teammates.



Our solution has two parts, classification and segmentation.


**Classification:**
This part is used to classify whether an image in related with pneumothorax or not.
Our model is a multi-task model based on unet with a branch for classifying. 
Data: all data
Cls loss: BCE + focal loss
Seg loss: BCE 
Augmentation: hflip, scale, rotate, bright, blur
Backbone: seresnext 50, seresnext101, efficientnet-b3
Ensemble: stacking


**Segmentation:**
There are two models used to segment, unet and deeplabv3.
Data: data with pneumothorax
Loss: dice loss
Augmentation: same as classification
Backbone: seresnext50, seresnext101, efficientnet-b3, efficientnet-b5
Ensemble: average


I’m cleaning source code, and will update it soon.

update
code: https://github.com/yelanlan/Pneumothorax-Segmentation-2nd-place-solution
