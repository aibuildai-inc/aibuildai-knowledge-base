# 2nd Place Solution

Competition: hubmap-organ-segmentation
Rank: #2
Source: https://www.kaggle.com/c/hubmap-organ-segmentation/discussion/354857

Big thanks to the organizers for the great competition!
Also thanks to all competitors for sharing their experiments here, especially @hengck23

# Models

In this competition the heavy encoders and larger resolution worked better. I've used 3 CNN encoders (efficientnet_b7, convnext_large, tf_efficientnetv2_l) and 1 transformer (coat_lite_medium). Coat performed the best as single model, but ensemble with CNNs scored more. Also tried few versions of swin v1 and v2, but it performed worse.

All models trained on 3 input resolutions: 768 * 768, 1024 * 1024, 1472 * 1472 with 5 folds.

Models also trained to predict organ and pixel_size. I think these aux outputs help to train more robust model. pixel_size calculated for resized input resolution and changed during training augmentations.

# Augmentations

random cropping/padding
scaling
rotating
flipping
color changing
blur/noise
saturation/brightness/contrast
elastic

# External data

External data helped a lot here. I've download some HPA data with this notebook https://www.kaggle.com/code/carnozhao/hpa-data-download and picked some images manually from sources posted here and previous Hubmap and Panda competitions. 

# Pseudo labeled

All external data pseudo-labeled using ensemble of initial models. 
Also training data was pseudo-labeled and in 30% used as ground truth for training.

# Color transfering

Training images recolored using this great notebook https://www.kaggle.com/code/gray98/stain-normalization-color-transfer with 3 different target images and used with 15% chance instead of original during training.

# Validation

Test-time augmentations used on validation (flip, crop, padding) + external data also separated on folds and used as validation to get best checkpoints. 


Inference notebook: https://www.kaggle.com/code/victorsd/2nd-place-inference/notebook?scriptVersionId=106240458
