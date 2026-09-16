# 3rd place solution - ResNet pretraining

Competition: mayo-clinic-strip-ai
Rank: #3
Source: https://www.kaggle.com/c/mayo-clinic-strip-ai/discussion/358187

Thanks to Mayo Clinic and Kaggle for this competition. I enjoyed this competition a lot. Here I would like to describe my solution in short.

# Pre-training
I think this is the most impactful process in my solution. Because the model weights trained by ImageNet would not be suitable for this task, I used "other" data to pretrain networks.

I separated "other" images into tiles, used the labels of "Unknown" or "Other", and pretrained ResNet152 models with ImageNet weights for 40 epochs. Then I trained the model to predict CE/LAA labels by using training dataset starting with pretrained weights.

In fact, the single pre-trained ResNet model got 0.65726 in Private Score, which is the best score among my submission.
(I did not select the best model as final submission. I selected  ensembled models with other no-pretrained methods instead, as described below.)
This is why I think pre-training was the most important process in my solution.

# Data

 - tiled into 512x512x3 channels and selected 16 instances for each image
 - saved as tiff format (LZW compression)

# Loss
 Because the labels of the training data are imbalanced, I used binary cross entropy with balanced class weights.
 
# Models
 ensemble of the following 3 methods
  * Resnet152 with pre-training
  * EfficientNetB0 without pre-training
  * Xception without pre-training

Thanks for reading.
