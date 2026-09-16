# 11th place solution [0.4796 private LB]

Competition: open-images-2019-instance-segmentation
Rank: #11
Source: https://www.kaggle.com/c/open-images-2019-instance-segmentation/discussion/111351

I would like to thank the competition organizers and all the competitors! 

Here's my brief solution writeup:

## 1. Dataset
- No external dataset.
I only use FAIR's ImageNet pretrained weights for initialization, as I have described in the Official External Data Thread.
- Class balancing.
For each class, images are sampled so that probability to have at least one instance of the class is equal (1/300) across 300 classes. One instance is randomly picked from an image to train the segmentation network described below.

## 2. Pipiline and Models
A two-stage pipeline with detection and single-instance segmentation networks is employed.
- Detection Model.
The detection baseline model is Feature Pyramid Network with ResNeXt152 backbone with modulated deformable convolution layers. (see [my post at the detection track](https://www.kaggle.com/c/open-images-2019-object-detection/discussion/110953)). 

- Segmentation Model.
The segmentation model is ResNet152-C4 with two upsampling layers and two U-net-like skip connections. 

Each instance is cropped from the image based on:
1) At training time: the ground truth bounding boxes.
2) At inference time: the bounding boxes detected by the (ensembled) detection model including the parent classes.
The cropped images are resized to (320, 320). The output mask resolution is (160, 160).

The models and training pipeline are developed based on the maskrcnn-benchmark repo.

## 3. Training
The training conditions are optimized for single GPU (V100).

- Detection Model.
The detection model has been trained using 500-class box labels and eight models are ensembled (0.597 private LB at object detection track).

- Segmentation Model.
The segmentation model has been trained for 1.8 million iterations and cosine decay is scheduled for the last 0.2 million iterations. Batchsize is 8 and batchnorm layers are used.

## 4. Ensembling
- Two models ensembling.
Two segmentation models with different image sampling seeds are ensembled with  and without horizontal flip. The output heatmaps are averaged.
- Results.
Model Ensembling improved private LB score from 0.4740 (single segmentation model) to 0.4796.
