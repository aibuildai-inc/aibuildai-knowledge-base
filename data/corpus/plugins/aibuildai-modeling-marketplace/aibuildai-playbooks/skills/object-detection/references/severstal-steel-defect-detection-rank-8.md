# 8th Place Solution

Competition: severstal-steel-defect-detection
Rank: #8
Source: https://www.kaggle.com/c/severstal-steel-defect-detection/discussion/114696

Congratatulations to the winners, and thanks to kaggle, competition sponsor and kernel contributors who shared their insights, it helped us a lot.

Models
We ensembled FPN-B0,B1,B2,B3,B4,Seresnext50. UNET-Seresnext50,Resnet34. Custom Attention Unet-B0,B1

Augmentations
Only Flipping and Random Brightness, Random Gamma, Random Contrast

Training
Progressive Learning was our primary approach started from 256x256 upto complete size.
256x256 were trained with encoder frozen for faster convergence - only batchnorm layers of encoder were unfrozen. In the end some of the models were over-fitted on complete train data(Stopped Early). 

Post - Training
Used Triple Thresholding ( Magic element) and only flipping as TTA.
