# 6th place solution [0.6023 private LB]

Competition: open-images-2019-object-detection
Rank: #6
Source: https://www.kaggle.com/c/open-images-2019-object-detection/discussion/110953

First of all, I would like to thank the competition organizers and all the competitors!
This was my first Kaggle competition and I really had a great time here 😀 

Here's my brief solution writeup:
## 1. Dataset
- No external dataset.
I only use FAIR's ImageNet pretrained weights for initialization, as I have described in the Official External Data Thread.
- Class balancing.
For each class, images are sampled so that probability to have at least one instance of the class is equal across 500 classes. For example, a model encounters very rare 'pressure cooker' images with probability of 1/500. For non-rare classes, the number of the images is limited.

## 2. Models
The baseline model is Feature Pyramid Network with ResNeXt152 backbone.
Modulated deformable convolution layers are introduced in the backbone network.
The model and training pipeline are developed based on the maskrcnn-benchmark repo.

## 3. Training
- Single GPU training.
The training conditions are optimized for single GPU (V100) training.
The baseline model has been trained for 3 million iterations and cosine decay is scheduled for the last 1.2 million iterations. Batch size is 1 (!) and loss is accumulated for 4 batches.
- Parent class expansion.
The models are trained with the ground truth boxes without parent class expansion. Parent boxes are added after inference, which achieves empirically better AP than multi-class training.
- Mini-validation.
A subset of validation dataset consisting of 5,700 images is used. Validation is performed every 0.2 million iterations using an instance with K80 GPU.

## 4. Ensembling
- Ensembling eight models.
Eight models with different image sampling seeds and different model conditions (ResNeXt 152 / 101, with and without DCN) are chosen and ensembled (after NMS).
- Final NMS.
NMS is performed again on the ensembled bounding boxes class by class. IoU threshold of NMS has been chosen carefully so that the resulting AP is maximized. Scores of box pairs with higher overlap than the threshold are added together.
- Results.
Model Ensembling improved private LB score from 0.56369 (single model) to 0.60231.
