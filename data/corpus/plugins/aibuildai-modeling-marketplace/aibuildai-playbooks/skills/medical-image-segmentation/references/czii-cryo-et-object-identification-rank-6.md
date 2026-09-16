# 6th Place Solution

Competition: czii-cryo-et-object-identification
Rank: #6
Source: https://www.kaggle.com/c/czii-cryo-et-object-identification/discussion/561518

First of all, I would like to express my sincere gratitude to the competition organizers for hosting this exciting challenge. Thanks to their efforts, I had the opportunity to learn a lot and improve my skills. I also appreciate the Kaggle community for their insightful discussions and valuable resources.
In particular, I found @hengck23's discussions and notebooks extremely helpful. The insights and contributions were instrumental in shaping my approach to this competition.

# Pipeline Overview
1.	Use MONAI’s `sliding_window_inference` to divide the input volume into fixed-size subvolumes and perform inference with 10 different models. Each model predicts a map for each particle type, indicating the locations of particles.
2.	Average 10 prediction maps for each particle.
3.	Binarize the averaged map using a threshold specific to each particle.
4.	Apply connected component analysis to the binarized volume and calculate the centroid of each connected component as the particle’s location.



# Models
I utilized a 2.5D UNet based on [this notebook](https://www.kaggle.com/code/hengck23/3d-unet-using-2d-image-encoder), as well as MONAI’s 3D UNet and SegResNet. For the 2.5D UNet, timm's EfficientNet-B2, EfficientNetV2-B2, ConvNeXt-Nano, and ResNet34d were used as encoders. While the 3D UNet and SegResNet initially struggled to achieve good performance using only the training dataset, they reached a similar level of accuracy as the 2.5D UNet after pretraining on simulated data, which I will describe later.
To improve model diversity, I trained a total of 10 models with slight variations in pretraining strategies, data augmentation techniques, and other hyperparameters. For the final submission, I used models trained on the entire training dataset.

# Training
## Preprocessing
As a preprocessing step, I first removed outliers and then applied min-max normalization. The input volumes were cropped to a fixed size of 64×128×128.

## Particle Mask
For the training labels, I generated binary masks centered around each particle’s location. Specifically, I set the region within radius × 0.5 to 1 and all other areas to 0. I created these masks separately for each particle.

## Loss Function
Initially, I experimented with BCE and Focal Loss, but these loss functions resulted in extremely slow convergence, likely due to the severe class imbalance. I then tried Dice Loss and Tversky Loss, which significantly accelerated training. However, these loss functions caused the model’s predictions to become overly binary, outputting only extreme values of 0 or 1. 
Since my pipeline ensembles prediction maps by averaging them and then applies a threshold for binarization, such prediction values were undesirable. To address this, I used FocalTversky++ loss as proposed in [this paper](https://arxiv.org/pdf/2111.00528). By using this loss, the model outputs prediction maps that reflect its confidence.



## Pretraining with Simulated Data
I generated my own simulated data using [polnet](https://github.com/anmartinezs/polnet) and used it to pretrain some of the models. To enhance the model’s ability to distinguish particles, I designed the simulated data to include particles with shapes similar to those of the target particles in this competition.



# Inference
For inference, I used MONAI’s `sliding_window_inference` with an overlap ratio of 0.25. Since predictions near the edges of subvolumes tend to be less stable, I discarded the outermost 8% of predictions to improve reliability.
I felt that one of the key challenges in this competition was achieving fast inference. To optimize speed, I first converted the models to TensorRT in a separate notebook before submission. During inference, I loaded the pre-converted TensorRT models and leveraged parallel processing with two T4 GPUs.

# Post Processing
Once the ensembled prediction maps were obtained, I binarized them using particle-specific thresholds. These thresholds were initially estimated through cross-validation and later fine-tuned based on leaderboard.
After binarization, I performed connected component analysis on the binary map and extracted the centroid of each connected component as the particle location.

# What Did Not Work
- Two-stage model
- Use of tomogram data other than denoised

# Code
- https://github.com/uchiyama33/czii-6th-place
- https://www.kaggle.com/code/tomoon33/czii-submission-6th-place
