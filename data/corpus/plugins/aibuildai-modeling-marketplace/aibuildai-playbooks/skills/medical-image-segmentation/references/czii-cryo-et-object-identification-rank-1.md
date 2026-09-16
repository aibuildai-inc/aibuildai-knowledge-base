# 1st place solution [segmentation with partly U-NET and ensembling part]

Competition: czii-cryo-et-object-identification
Rank: #1
Source: https://www.kaggle.com/c/czii-cryo-et-object-identification/discussion/561510

Thanks to kaggle and everyone involved for hosting this exciting competition. It was a great learning experience and it was very interesting to see how much of our computer vision experience could also be applied to 3D imaging. Thanks to @bloodaxe for this great team experience. 

## TLDR

The solution in an ensemble of segmentation (3D Unets with ResNet & B3 encoders) and object detection models (SegResNet and DynUnet backbones) from [MONAI](https://github.com/Project-MONAI/MONAI). We also used MONAI for augmentations, and exported models via jit or TensorRT, which gave 200% speedup increase and enabled us to have a slightly larger ensemble. We did not use any external or simulated data!

This post covers the segmentation based approach and ensembling. For object detection part see @bloodaxe writeup: [1st place solution [Object Detection Part]](https://www.kaggle.com/competitions/czii-cryo-et-object-identification/discussion/561440)

## Cross validation

For segmentation approach 7 folds were used, simply by splitting by experiment. Using mean of f4-score of all 7 folds had some good correlation with LB. During model training I optimized individual class thresholds at the end of each epoch by simple grid-search on the validation experiment. After all 7 folds were trained we could re-calibrate thresholds, by taking OOF predictions. And fitting the threshold for one fold on the predictions of the other 6. Then we average the resulting f4-curves and take the best threshold.

## Data preprocessing/ augmentations

3D images were normalized by standard normalization, i.e. for each 630x630x184 image, we substract mean and devide by standard deviation before splitting the images into patches.
Since models are trained from scratch, augmentations were essential to prevent overfitting. 
We used RandomCrop, Flip on each axis, and rotation, which all are available with MONAI. Additionally I used my own implementation of MixUp which was highly effective to train longer and prevent overfitting.

## Model

Modelling was quite a ride in this competition. I started with simple UNET with having 3D gaussian balls as segmentation target. For diversity I also tried [object detection example from MONAI](https://github.com/Project-MONAI/tutorials/tree/main/detection) and realized its working very well out of the box. But when analyzing the different output feature maps and trying to isolate the perfomance gain over gaussian heatmap based segmentation I realized where the advantage was and adjusted my segmentation model accordingly. I learned that 

- the penultimate feature map has a higher accuracy than the last one, which is surprising at first
- gains from box regression are negligable, as particles from the same type have mostly same size anyways.

Hence its sufficient to have a pixel-wise loss on the penultimate feature map output. I.e. use a **partly** UNET. The gaussian heatmap is not needed when suppressing background with a low class weight, and just use single pixels as targets. Using the same approach on lower level outputs (= deep supervision) is possible, but does not provide much gain. Input for the segmentation models are 96x96x96 image patches and the loss is calculated on the 48x48x48 output. 



In general, we observed that relatively small models work really and the design of loss is the most important aspect. We used MONAIs [FlexibleUnet](https://docs.monai.io/en/stable/networks.html#flexibleunet) with backbones resnet34 and efficientnet-b3. Six checkpoints of this architecture would finish under 2h and score 7th place on LB.

## Training procedure

We used 7 classes (incl background) and weighted CrossEntropy as loss. Notably keeping beta-amylase as a class although it is not scored is quite helpful as model learns to differentiate beta-galactosidase from it. To account for low number of positive pixels, positive pixels are weighted by 256 and background has weight 1. Models were trained with a cosine learning rate schedule with peak LR of 0.001, mixed precision and an effective batch size of 32 samples. Training is based on Random crops and for validation the single experiment image is divided into patches and stored in RAM. 

## Ensembling

Ensembling was very challenging, as our two approaches are quite diverse. While in theory predictions from the segmentation models can be ensembled with feature map outputs of the object detection model before runnning the object detection postprocessing, in practice those feature maps have a very different distribution due to difference architectures and loss functions. 



We were eager to find an elegant way to fix this scaling issue as we saw the potential of a possible ensemble. The scaling of our best submission works as following. To combine predictions A with predictions B, for each class sort all pixel values for A and B and replace values of B with the corresponding values of A of same rank. In code: 



This results in both predictions having the same distribution, and hence we could simple blend the feature maps before performing object de
tection task.



## What did not work
- Using supplemental data (external or simulated)
- Other augs
- Other losses (Tversky, Dice)

Thanks for reading. 

Edits:
training code: https://github.com/ChristofHenkel/kaggle-cryoet-1st-place-segmentation
inference kernel: https://www.kaggle.com/code/christofhenkel/cryo-et-1st-place-solution?scriptVersionId=223259615
