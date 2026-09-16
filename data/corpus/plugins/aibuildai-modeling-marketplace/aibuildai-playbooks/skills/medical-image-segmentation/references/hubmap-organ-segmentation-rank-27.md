# 27th Simple Solution

Competition: hubmap-organ-segmentation
Rank: #27
Source: https://www.kaggle.com/c/hubmap-organ-segmentation/discussion/355213

First of all, thank the organizers for hosting this competition. 
We also thank @hengck23 for his discussions and notebook.

**Models**
Coat_small+daformer,size=768,folds=5,swa
Coat_small+daformer,size=768,folds=5,External data , swa 
Coat_lite_medium+daformer,size=768,folds=5,swa
Coat_lite_medium+daformer,size=768,all data,swa
Coat_small(level5)+daformer_conv1*1,size=1152, folds=5, swa
pvt_v2_b4+daformer_conv3x3,size=768,folds=5,swa
pvt_v2_b4+daformer_conv3x3,size=768,all data,swa
pvt_v2_b4+smp_unet,size=768,folds=5,swa
pvt_v2_b4+smp_unet,size=768,all data,swa

**Augmentations**
Random flip, rotate, noise, HueSaturationValue, affine..

**Data Processing**
We have selected three methods of training data:
The original data is divided into training data and validation data for training. 
The full data is trained. 
The external data is labeled and then added to the original data for training

**Inference**
Ensemble
Coat_small+daformer:fold0 fold3
Coat_lite_medium+daformer: fold2 fold3 fold4 all_data
Coat_small(level5)+daformer_conv1*1 :fold0_1152
pvt_v2_b4+daformer_conv3x3:fold1,fold2,fold3,fold4,all_data,
fold3_384,fold2_384,fold3_1024
pvt_v2_b4+smp_unet:fold0,fold2,all_data

TTA
Flip,Rorate

Organ Threshold
 'Hubmap': {
        'kidney'        : 0.35,
        'prostate'      : 0.25,
        'largeintestine': 0.3,
        'spleen'        : 0.3,
        'lung'          : 0.1,
    },

**Work**
SWA, TTA, Adjust the threshold, model fusion, Multi-input size fusion, Augmentation(Random flip, rotate, noise, HueSaturationValue, affine)
**Not Work**
 Augmentation - rgbtogbr and cutout,
 External data and Stain normalization.
