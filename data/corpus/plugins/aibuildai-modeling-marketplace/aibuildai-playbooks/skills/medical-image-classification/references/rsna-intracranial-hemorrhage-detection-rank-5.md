# 5th place solution (with code).

Competition: rsna-intracranial-hemorrhage-detection
Rank: #5
Source: https://www.kaggle.com/c/rsna-intracranial-hemorrhage-detection/discussion/117232

Congrats to all winners,  

Congrats to my teammate @tarobxl achieving GM tier and @anjum48 for his master tier.  

On behalf of the team, I would like to make the writeup. 

# Image Preprocessing  
We have three types of preprocessing data.  
1. Imaging with multiple windows.  
    We use three windows to construct RGB image. Each channel is corresponded to a window. 
    ```
    'brain': [40, 80],
    'bone': [600, 2800],
    'subdual': [75, 215]
    ``` 

2. Imaging with multiple windows then crop.  
    Same as (1), we crop and keep the only informative part.

3. Imaging with spatially adjacent.  
    We use only one window [40, 80] for preprocessing. To construct RGB images, we use metadata to know the spaitally adjacent. Let say to construct RGB of slice St, we take:  
    R = St-1, G = St, B = St+1. 

    Finally, we crop and keep only informative parts as same as (2). 
    Please refer this kernel for more detail: 
    https://www.kaggle.com/anjum48/preprocessing-adjacent-images-and-cropping



# Data Preprocessing  
First, we remove the overlapped patients between train and test. This part may be the reason for the shakeup since we estimate that the shakeup score is in a range of 0.001 - 0.002.

In each fold, we do random sampling such that the number of positive patients ar balanced to the number of negative patients. This step helps to have the correlation between CV and LB, and stable as well. 

# Modeling 
We train 5 Folds splitted by patients. The models and their performance on stage 2:   

|No. | Model         |      Data     |  Before PP | After PP|
|---|----------     |:-------------:|-------------------------|---:|
|1. | Resnet18      |  (1)          | 0.060     |  0.054    |
|2. | Resnet34      |  (1)          |   -       |  -        |
|3. | Resnet50      |  (1)          |    0.058  | 0.052     | 
|4. | Resnet50      |  (3)          |    0.054  | 0.051     |
|5. | Densenet169   |  (2)          |    0.055  | 0.049     | 
|6. | InceptionV3 + Deepsupervision | (1) |    0.060 | 0.053 |
|7. | EfficientNet-B0 | (3)         |    0.054      |  0.051 |
|8. | EfficientNet-B3 | (2)         |    0.055     | 0.050 |
|9. | EfficientNet-B5 | (3)         |    0.048  |   0.048 |  

PP =Post-processing. 

We have three pipelines, the following is mine which is used to train the model `No. 1,2,3,4,5`. 

* Optimizer: AdamW 
* Image size: 512x512 
* Stages: 
    * Warmup: Freeze the backbone, train the FC only.  
        * LR: 0.001
        * num_epochs: 3 

    * Warmup: Unfreeze the backbone, train all the model.  
        * LR: 0.0001
        * num_epochs: 20 
        * scheduler: ReduceLROnPlateau, patience = 0. 
        * EarlyStoppingCallback: patience = 3. 
* Augmentations: 
    ```python
    Resize(*image_size),
    HorizontalFlip(),
    OneOf([
        ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03),
        GridDistortion(),
        OpticalDistortion(distort_limit=2, shift_limit=0.5),
    ], p=0.3),
    ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=10),
    ``` 

* TTA: Normal + HFlip. 

By this pipeline, I finish the training at around 8-10 epochs. The deep models (SEResnext50, resnet101, etc) do not work well. Training with longer epochs (upto 25) leads to be overfitted. 


# Post-processing  
We leverage metadata and use H2O to build a model for post-processing. 
More details will come up by  @tarobxl. 

# Stacking  
First, the do post-processing for each prediction of each model.  
Second, we use the stacking pipeline designed by magician @mathormad. 
Please upvote this topic: 
https://www.kaggle.com/c/imaterialist-challenge-fashion-2018/discussion/57934  

Update:   
Stacking pipeline is shared at:  
https://www.kaggle.com/mathormad/5th-place-solution-stacking-pipeline  
Dont hesitate to upvote it.


# Code 
My pipeline code is published at:   
https://github.com/ngxbac/Kaggle-RSNA  
The model checkpoints, graph, training processes are recorded by wandb.
https://app.wandb.ai/ngxbac/Kaggle-RSNA 

@anjum48 's pipeline: 
https://github.com/Anjum48/rsna-ich 

@mathormad's pipeline to train InceptionV3 + Deepsupervision:  
https://github.com/triducnguyentang/RSNA

@tarobxl 's post-processing code: 
https://github.com/tiendzung-le/Kaggle-RSNA-5th-place-Solution
