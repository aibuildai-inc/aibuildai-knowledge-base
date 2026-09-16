# 26th place solution | Recursive Gated Convolutions + Lung Tiling

Competition: hubmap-organ-segmentation
Rank: #26
Source: https://www.kaggle.com/c/hubmap-organ-segmentation/discussion/354671

*Dedicated to the Armed Forces of Ukraine!*
*Glory to the Armed Forces of Ukraine for the opportunity to provide science and education for the benefit of Ukraine!*

----

Hello everyone from the complex team! 
First of all, I would like to give my huge thank you to the Kaggle team for hosting such an amazing competition. 
Every time, I feel honored to be a part of such outstanding research.
I want to give a big thank you from our team to @hengck23. We have learned so much from your notebooks :)
It is always a pleasure to have this great of a competitor from whom you can learn!


Our submission is a result of a few stages. We don't mainly do any pre-processing of the data. Although, at the very end of the competition, we found a lack in annotations that could have caused a slight weakening of masks and added noise to the training [read about it at the end of the overview]

 
# Overview

After many experiments with training all organs with one model, we gave up this idea. The reason was we didn't see any improvement with it after 0.77. 
Thus, we decided that the model needed contextual information about the organ type. The first experiments were set up with additional aux classification head for classifying each organ type. However, this became a false path after all; the models couldn't converge, and scores were not getting better.

Then, we decided to train separate models for the lung and the rest of the organs together. We adopted this strategy and proceeded with training the models in different conditions:
- we use full masks resized to 768 for ['kidney', 'largeintestine', 'lungs', 'prostate', 'spleen'], + some models were trained with lung excluded from the train set
- for lungs, we chose only type1 masked lungs [Lung Segmentation Section] and pre-processed the mask batch by resizing it first to 1024 and then tiling with kernel size 512 and stride 2 

We ensemble 6 models with different weights applied (5 folds per model) based on the best local cv scores. Plus, only one model for the lungs.



# Segmentation 



Before training, we resized tissue slices to 768 and trained using strong augmentation. 
We find that intense color shifts and dropout noise produce good results. 

```python
_transforms = A.Compose([
    A.VerticalFlip(p=0.5),
    A.HorizontalFlip(p=0.5),
    A.RandomRotate90(p=0.5),

    A.RandomBrightnessContrast(brightness_limit=(-0.02, 0.02), contrast_limit=(-0.02, 0.02), p=0.25),
    A.RGBShift(r_shift_limit=(-25, 25), g_shift_limit=(-20, 20), b_shift_limit=(-20, 20), p=0.5),
    A.HueSaturationValue(hue_shift_limit=(-15, 15), sat_shift_limit=(-15, 15), val_shift_limit=(-15, 15), p=0.5),
    A.CoarseDropout(max_holes=20, min_holes=7, p=0.5),
    A.ShiftScaleRotate(shift_limit=0, scale_limit=(-0.2, 0.2), rotate_limit=0, p=0.5),
    A.OneOf([
        A.GridDistortion(num_steps=5, distort_limit=0.05, p=1.0),
        A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=1.0)
    ], p=0.5),
])
```
----
**The resulting pipeline is a model zoo :)**

We use pretrained ConvNext and HorNet backbone architectures. For the decoder block, we only use UPerNet.

- *Inside: After the competition, we find that HorNet with recursive gated convolutions for high-order spatial interactions and large receptive field (the 7x7 conv and global filter) gave significantly stable results for all organs!*

### ConvNext vs HorNet
Best HorNet Single Model Submissions



Best ConvNext Single Model Submissions



# Lung Segmentation | Tiles

We used the tiling approach for lung training, which we found to be very robust. The images were first resized to 1024 size. After, the image batch got tiled the convolution way with 0.5 overlaps between neighboring tiles. 
We also try randomly sampling 6 out of 9 tiles for data variability per mini-batch. Thus, we result in a bigger batch size with smaller amount of tiles per tissue slice. However, the LB standing didn't change. Indeed, it stayed the same.



For the inference, we rescale the images normalized to tissue slice thickness of size 1024 and use full images for predictions.



### type1 masked lungs
After reviewing all the lung masks and corresponding images, we concluded that several annotators might have done annotations. 

Thus, we decide to cluster the lung set into two groups with similar annotations styles. We find the following:


***type1_anns** = [10488, 11064, 11629, 1220, 12827, 15067, 15329, 1731, 1878,
              20563, 24782, 26480, 27232, 2793, 28052, 28189, 28429, 30084,
              31800, 32151, 4301, 4412, 4776, 5086, 5552, 8231, 8343, 9387,
              9450, 12452, 14388, 25516, 30394, 5777, 686]*

***type2_anns** = [12476, 15124, 23252, 25945, 30500, 31571,
              7359, 8151, 127, 13189, 16564, 29610, 31139]*

***type3_anomaly_anns** = [30084, 31800]*




# Post-Processing | Refine Mask

Here we tried fixing the morphological structure for kidney masks. Instances of kidney masks are pretty separate from each other. Thus, we expected not to have many problems with watershed segmentation. If the instances result in touching points at the inference stage, we apply binary opening with disk size 5 and opening with disk size 10 operations to remove the extra connected parts if present. 


# Adaptive Thresholding

This is one of the essential parts of the prediction side. We notice that hand-picking thresholds might result in some inconsistencies for some predictions.
Thus, we present the adaptive way for thresholding a probability map. We find that models indeed can produce an accuracy measure: 
- We calculate the mean of the models' prediction mask and set it as our threshold. We additionally divide it by 1.75 constant to lower the threshold step. We find that this constant is as good for kidney predictions as it is suitable for lungs.
- Additionally, assuming every mask can't be empty, we utilize a linear descent in the threshold by the factor of 0.2 if any segmentation is found to be empty.

# LB Scores

Private/Public: 0.81/0.81 (many models ensemble + thresholding strategy introduce good metric stability :))

### Ideas we didn't try:
- Pseudo-labeling/SSL (unfortunately, we didn't have time to try this approach; it seems to be a game changer for many teams)
- Cleaning masks before training (hole filling)



# **Glory to Ukraine! Glory to Heroes! 🇺🇦**
