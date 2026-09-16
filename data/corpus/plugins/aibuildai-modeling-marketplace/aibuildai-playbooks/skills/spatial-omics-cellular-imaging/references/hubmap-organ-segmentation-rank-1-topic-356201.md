# 1st place solution

Competition: hubmap-organ-segmentation
Rank: #1
Source: https://www.kaggle.com/c/hubmap-organ-segmentation/discussion/356201

First of all, I’d like to express my gratitude to kaggle and the competition hosts for holding such an amazing competition.
I’d like to thank MMSegmentation developers too. MMSegmentation is really nice and useful tool. Also great thanks to @ayuraj who incorporated a hook of Weight & Biases into MMSegmentation, which helped a lot to manage training history.
 
In the contest, I learned a lot from @hengck23. Thank you so much for sharing knowledge with us:)
 

### **Overview**
I used the following methods.
・ End-to-end MMSegmentation(with customization)
・ Ensemble of SegFormers
・ Used group normalization and trained with batch_size=1 in Colab
・ Hand-labeling of lung from scratch
・ External dataset and pseudo labeling.
 

### **DataSet**
Used dataset:
・ HPA images [351 images]
・ HPA images stained with H&E or PAS [351 images x 4 stain patterns]
・ external spleen [39 images + 39 images x 4 stain patterns]    *from a single image
・ external lung [73 images]    *from a single image
 
**(Lung)**When I noticed that unstable results were from lung prediction, I imagined  hand labeling is better than predicting lung by low threshold, which might lead to unstable and unconfident results. So  I researched how lung alveolus/alveoli look alike and hand-labeled lung images from scratch. Since there is no information on how to annotate alveolus/alveoli anywhere, I made about 7 annotation sets where potential alveolus was gradually annotated and probed which one is the best. Then I created an external dataset with pseudo labels and with the best threshold. After training with it, the best threshold for lung increased to 0.60, which gave the best result ever. I guess good modeling is better than hand labeling since there are those who got 0.11/0.22 in public/private. 
**(Spleen)**Used pseudo label for an external dataset
**(Prostate/ Largeintestine)**Used pseudo label for the HPA dataset.
I had a feeling that my models underfit HPA dataset because these dataset didn’t put weight on HPA datasets in terms of colors. The torchstain was used for the stain tool.
 

### **Models**
Ensemble of the following models:
・ 1x SegFormer mit-b3, image_size: 1024x1024
・ 2x SegFormer mit-b4, image_size: 960x960
・ 1x SegFormer mit-b5, image_size: 928x928
・ 2x SegFormer mit-b5, image_size: 960x960
Different pretrained models, seed and additional stain dataset were used in some models. The reason all models are SegFormer is I chose models only from MMSegmentation and it worked well.
*Best single model was mit-b4(private: 0.82821).


### **Traning**
I used google colaboratory. Since GPU resources are limited, I trained segformer with batch_size=1 with group normalization.  In my experiment, it was better than batch_size=4 with batch normalization in both CV and LB. I chose 2 class rather than 6 class segmentation because it showed better results consistently, especially in lung. In 6 classes, the area of the lung tends to shrink and even if the area was equivalent to 2 classes, LB lung score was worse.

settings:
・ norm: group normalization(num_groups=32) for decoder
・ num_classes:2
・ loss_decode: CE:LovaszLoss=1:3
・ steps: 45000
・ optimization: AdamW with lr=0.00006 

augmentations:
・ Dataset image size: 2000x2000
・ Downscale for prostate
・ Crop/ Flip/ RandomRotate90/ HueSaturationValue/ RandomGamma/ RandomBrightness
・ (mmseg)Resize to (480~1600, 480~1600)  * such as (480, 1600), (1024,1024), (1500,480)
・ (mmseg)RandomCrop(crop_size=model input size)


### **Inference**
・ Whole image and ignored pixel_size
・ TTA: horizontal flip and different scales (approximately x0.8, x1.0, x1.2) .
・ Thresholds for each organ were  kidney: 0.3 | large intestine: 0.2 | lung: 0.6 | prostate: 0.3 | spleen: 0.5
 

### **Results**
all		private: 0.83562	public: 0.82716 
kidney		private: 0.16940		public: 0.12175
large intestine	private: 0.09005		public: 0.05730
lung		private: 0.21586		public: 0.10963
prostate	private: 0.17784		public: 0.14772
spleen		private: 0.18246		public: 0.17515
 
 
### **About license**
Pretrained SegFormer and its encoder “mit” are unavailable for commercial use.  It seems that other winners also use this model. However, I got an answer from the host, which permits the use of SegFormer in this competiion. The reasons are:
1. Since competition hosts are non-profit, the commercial restriction isn’t applied.
2. The model was Apache 2.0 license at the competition launch and  the license was changed during the competition.
