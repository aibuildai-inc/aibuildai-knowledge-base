# 7th solution

Competition: blood-vessel-segmentation
Rank: #7
Source: https://www.kaggle.com/c/blood-vessel-segmentation/discussion/475964

First and foremost, I'd like to extend my gratitude to Kaggle and the competition organizers for creating such a compelling event. Despite joining the contest relatively late, I was able to quickly get up to speed thanks to insightful discussions by @hengck23 , the informative videos by @yoyobar , and the vibrant exchanges within the community.

## Overview

I employed a hybrid approach for the model, utilizing 2.5D images as input. The architecture combines a 2D Unet framework with 3D convolutional layers. From my research and the community's insights, it seemed that a full 3D model might offer superior results compared to 2D models. However, due to the high computational costs associated with 3D models, I choose a blend of 3D convolutions within a 2D Unet structure, striking a balance between efficiency and performance.

## Data Preparation

### Generating 3D Rotational Slices

I augmented the dataset with 3D rotation. The process begins by assembling the images into a 3D volume, followed by rotating two axes and extracting slices along the remaining axis. The rotation angles used are as follows:

```python
rotation_angles = [
    [10, 10], [10, -10], [-10, 10], [-10, -10],
    [30, 30], [30, -30], [-30, 30], [-30, -30],
    [45, 45], [45, -45], [-45, 45], [-45, -45]
]
```

Post-rotation, some slices exhibited increased areas of black background. To maintain data quality, I retained only those slices where the target segmentation was present and the black background constituted less than 50% of the slice area. 
#### Rotate data sample

### Pseudo Labeling

The pseudo labeling process involved:
1. Generating additional slices for kidney1 and kidney3 using the aforementioned technique.
2. Training a model with the augmented dataset.
3. Applying the model to pseudo label kidney2, followed by generating extra slices for it in a similar manner.

## Model

### Architecture

### Training and Inference Details

#### Training

The training setup was as follows:
1. The model takes a 3-channel 2.5D image as input and outputs a 3-channel prediction.
2. Normalize the input base on the std of each kidney
2. I used a combination of loss functions: BCEWithLogitsLoss, DiceLoss, and FocalLoss. The loss for each of the three channels was calculated separately, with the middle channel assigned a higher weight of 0.9.
3. Optimizer: AdamW
4. Scheduler: CosineAnnealingLR
5. Images were cropped to a size of 1024x1024 for processing.

#### Inference

For inference:
1. A single model was used for predictions.
2. The model operated at the original image resolution.
3. Similar to training, the input comprised 3-channel 2.5D images, with the output being a 3-channel prediction, primarily focusing on the middle channel.
2. Normalize the input base on the std of each kidney
4. Predictions were made along the x, y, and z axes, and the results were averaged.
5. Test Time Augmentation (TTA) included horizontal flipping.
6. Post-processing steps involved applying thresholds of 0.2, followed by a 3D closing operation.
