# 20th Place Solution - Two stage model

Competition: hpa-single-cell-image-classification
Rank: #20
Source: https://www.kaggle.com/c/hpa-single-cell-image-classification/discussion/239641

Thank you to the competition host for such an interesting competition. I had quite some fun and learnt a lot from this competition. Thank you fellow HPA competitors for sharing your ideas and solution. To the winners of this competitions, congratulations!

# Introduction
Like most competitors I used the segmentation algorithms provided by the organizers to segment the cells . The data provides weak positive labels, but an abundance of strong negative labels. The converse is true for the negative class. So I trained a two-stage model, the first on image level labels and the second on cell level labels predicted by the first stage.

# Stage 1
An ensemble of 6 efficientnet-b0 models to predict image level labels, trained with different random seeds but with the full datasets.

## Augmentation
- Random flip
- Random rotate 90
- Random brightness (independent in each RGBY channel)
- Random crop

## Training settings
- A slightly modified [balanced focal loss](https://openaccess.thecvf.com/content_CVPR_2019/papers/Cui_Class-Balanced_Loss_Based_on_Effective_Number_of_Samples_CVPR_2019_paper.pdf). Instead of assigning a weight to each class I calculated a different alpha for each class assuming a one-vs-all classification.
- cosine annealing with warmup
- 1e-2 learning rate
- 10 warmup epochs
- 90 epochs
- AdamW with 1e-3 weight decay

## Results on LB
The stage 1 model has a LB score of 0.47455 private, 0.48021 public with 8x TTA.

# Stage 2
In stage 2, I made two assumptions on the image level labels provided by the organizers.
1. Positive labels indicates at least one cell in the image with the target class.
2. Negative labels indicates the all the cells in the image do not contain the target class.

So I used the stage 1 ensemble to predict individual cell labels but I only keep the cell-wise labels for classes where their ground truth image label is positive. The cell labels for classes which have negative ground truth image label are set to 0.

For the negative classes, I predict negative_prob = 1 - max(other classes). I downloaded 1123 negative images from the extra HPA dataset and assign a negative class label of 1.

I then trained this stage 2 model with the following setting.

## Augmentation
- Random crop 10% off height and width
- Random flip
- Random rotate 90
- Random brightness (independent in each RGBY channel)
- With 50% probability, I multiply the red, blue and yellow channels with a random number between 0 and 1. This is motivated by the observation that some duplicate images have the same structure but different intensities between different channel.
- Crop or pad images to 224x224

## Hyperparameters
I didn't have time to try different hyperparameters but the settings I used for my solution is
- Cosine annealing with warmup
- learning rate 1e-3
- 10 warmup epochs, 30 epochs
- AdamW optimizers with 1-e4 weight decay (this is quite important as the cell-level models overtrain very quickly into NaNs on mixed precision without weight decays).

This model is very expensive to train as I had more than 500,000 examples. One epoch took 6 minutes on the latest gen GPU.

## Results of stage 2 training
The stage 2 training improve my score by about 0.3 in my best solution, others vary from 0.1-0.8 based on what I tried in stage 1 models. My best submission has a score of 0.50626 private 0.50738 public.

# Extra Data
I downloaded the extra public dataset provided by the organizers but I only selected from classes which had image-level mAP scores of less than 0.8 based on the validation set of my stage 1 model. I also downloaded 1123 negative (no target class) images.

# Choice of model
I noticed efficientnet-b0 performed the best in stage 1. Other models I tried were mobilenet v3 and resnet-50, resnet-101. Larger efficientnet models (I tried b1 and b2) didn't perform as well as b0.

# Post-processing
Unfortunately, I ran out of time towards the end of the competition and I wasn't able to try combining image level predictions with cell level predictions or trying something to deal with edge cells. (One thing I learnt was that time management is quite important in competitions like this.)

# Things that did not work
- Concatenate a global max pool and global average pool in the last layer of the stage 1 (image level) models.
- [DRS](https://arxiv.org/abs/2103.07246) in the global average pooling layer. Maybe there is some hyperparameter setting or something I got wrong.

# Lack of plots or image
It seems I cannot upload images to kaggle at the moment. I hope to update this post with plots or images in the future when that is possible.
