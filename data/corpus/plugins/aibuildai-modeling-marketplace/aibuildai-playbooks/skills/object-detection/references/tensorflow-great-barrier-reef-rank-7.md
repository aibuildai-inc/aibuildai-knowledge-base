# 7th Place Solution - Cascade RCNN+Tracking ( Public LB is not all you need.)

Competition: tensorflow-great-barrier-reef
Rank: #7
Source: https://www.kaggle.com/c/tensorflow-great-barrier-reef/discussion/307786

We sincerely thank the Tensorflow organization for hosting this awesome competition. Please allow me to send my best regards to my teammates for their dedicated efforts. 
This is the first detection competition for our team and it is a great honor for us to obtain the final score of Rank 7. 
The overall solution is based on Highly Customized Cascade RCNN with Tracking as a post-process method. Moreover, the good fortune and the robustness of the overall design play a vital role in this competition. 

# Summary
The main idea of our method is based on the stronger baseline **cascade rcnn model design** with lots of customized features, the **careful-crafted data augmentation strategy** and **object tracking post-procedure**. Our method is a single model approach and it does not rely on any ensemble strategy. 

## Data Split. [Update]  
At the beginning of the competition,  we split the dataset with a randomly split training set by video ids. 
We use train-val set split to select the appropriate model. After selecting an appropriate model design, We switch to the full trainset and perform the comparison on Public LB. 

## Stronger Baseline of Cascade RCNN.
The team has designed a customized Cascade RCNN baseline. The majority of model improvements are listed as follows. 
1. Stronger Backbone: ResNet -> ResneXt -> Res2Net -> CBNet; 
2. Enhanced FPN: FPN -> PAFPN; 
3. Customized Detection Heads: Cascade RCNN Head -> Double Head Cascade RCNN Head;
4. Loss function: Smooth L1 loss -> IoU Based Loss;

## Careful-crafted Data Augmentation Strategy.
1. Weak Aug: Flip, RandomBrightnessContrast, RGBShift, HueSaturationValue, Noise, CLAHE, Affine, Rotate
2. Strong Aug: Copy-Paste, Mosaic, AutoAugmentation V1 policy, Mixup, Cutout
3. MS Training and Testing: [0.8 * image_size, 1.2 * image_size]

## Object Tracking as Post Process. 
Inspired by @parapapapam (https://www.kaggle.com/parapapapam/yolox-inference-tracking-on-cots-lb-0-539), We adopt Norfair tracking as a post-process and gain a performance boost. Thanks for the kind sharing.

## Public LB is not all you need. 
Since our method is based on the above-mentioned methodology, the proposed method suffers from a significant performance drop in public LB. The team is in deep desperation when the rank of public LB goes to No. 1000. Thankfully, the proposed method demonstrates its robustness in the private LB. 

We have learned a lot from the competition, and our method can get an additional performance boost if we have more time on exploring model ensemble strategies. 

## Most Commonly Asked Questions: 
Thank you for everyone who shows great respect to this competition. Our implementation of Cascade RCNN are build on [1]. Since kaggle notebook is not friendly for MMDetection codebases (All custom codes turns to private_datasets... ) and some private code issues (Company limitations), we will clean up the codebase, construct a public notebook and share detailed configurations in the future. 
[1] https://github.com/shinya7y/UniverseNet
