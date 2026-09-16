# 6th place solution

Competition: siim-acr-pneumothorax-segmentation
Rank: #6
Source: https://www.kaggle.com/c/siim-acr-pneumothorax-segmentation/discussion/107743

First of all, we would like to thank SIIM and ACR for this interesting competition.

Our final solution based on EncodingNet (ResNets, 512 and 1024 size) and UNet (EfficientNet4, se-resnext50, SENet154 with 512, 640 and 1024 sizes).
Our models were trained and tuned independently before team merge. This gave us a lot of  diversity in solutions.

Best augmentations were related to crops and rotations. We didn’t use contrast and brightness transformations.
Loss: BCE+Dice (tried Focal, didn’t work)
Lower image sizes reduced score a lot so full sized model should be better.
Tricks:
classification on top of EncondingNet with heavy TTA (11 methods)
small segments of predicted masks was deleted

Also I noticed that Horizontal Flip augmentation reduced my local CV, but didn’t check it on LB.

Link to github:
https://github.com/yura03101995/siim_pneumo

UPDATE:
For validation we used 4 folds. Folds were generated using `StratifiedKFold(n_splits=8)` along `isPneumothorax`. For each `solo model` we did
4-fold-blend model. Augmentations we used

```
AUG = Compose(
    [
       HorizontalFlip(p=0.5),
       OneOf(
           [
              ElasticTransform(
                  alpha=300,
                  sigma=300 * 0.05,
                  alpha_affine=300 * 0.03
              ),
              GridDistortion(),
              OpticalDistortion(distort_limit=2, shift_limit=0.5),
           ],
           p=0.3
       ),
       RandomSizedCrop(min_max_height=(900, 1024), height=1024, width=1024, p=0.5),
       ShiftScaleRotate(rotate_limit=20, p=0.5)
    ],
    p=1
)
```


**Model1**:
**EncNet** (from pytorch-encoding) with **Resnet50** pretrained on Pascal dataset.
First, we trained 4-fold-blend model on 512 x 512 resolution with AUG.
Second, we trained 4-fold-blend model on 1024 x 1024 resolution with AUG .
Finally, we avereged two 4-fold-blend models.                   

**Loss**: Dice
**Augmentation**: AUG
**TTA**: Flip, Clahe and `np.arange(-20, 21, 5)` angles rotations (total 10 TTA)

**Model2-3**:
**Two Unet**-like models with **SEResnext50** and **SEResnet152** backbones were trained on
1024 x 1024 resolution. Each model also was averaged along 4 folds.

**Loss**: Dice     
**Augmentation**: AUG
**TTA**: Flip

**Model4**:
**Unet**-like model with **EfficientNetB4** backbone trained on **640 x 640** resolution.
Also were blended along 4 folds.

**Loss**: weighted Dice + BCE
**Augmentation**: AUG
**TTA**: no TTA

**Final model**: 
Average of **Model1**, **Model2**, **Model3** and **Model4**

**Classification threshold**: ~0.32
**Segmentation threshold**: ~0.375
