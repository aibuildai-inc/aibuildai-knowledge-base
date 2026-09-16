# 30 place solution writeup

Competition: human-protein-atlas-image-classification
Rank: #30
Source: https://www.kaggle.com/c/human-protein-atlas-image-classification/discussion/77285

First of all, thanks to Kaggle and HPA team for this interesting competition!. This is the most difficult and memorable competition since I start my journey as a Kaggle competitor. I learned a lot from this

Overview, our solution is so basic. I see there are some interesting points inside it. I hope they will be useful for you.  

1. Data  
We use RGBY and addition HPAv18. 
2. Preprocessing  
 - We were only able to train with 512x512 images. To save the disk space and loading time, we resize the HPAv18 image to 512x512.  
 - Upsampling rare classes: Any class which has a number of image less than 450, we upsample up to 450 with random rotation from (0, 360) 
3. Model  
 We use following model for final submission: 
 - SEResnext50 ( public: 0.604, private: 0.534)
 - SEResnet50 (public: 0.599, private: 0.534)
 - InceptionV3 (public: 0.585, private: 0.521) 
 - Resnet34 (public: 0.580, private: 0.508) 
4. Loss  
 In our experiment, weighted BCE loss performs best. Other combinations such as: FocalLoss, F1 
 does not work. We stuck at public 0.577LB for a long time when using it.  
 After that, we try a tricky loss method: Train the model with weighted BCE loss with even epoch, 
 and F1 loss with odd epoch. Then, we get 0.04 addition for each model and ensemble. 
5. Augmentation 
 - Rotation (0, 360) 
 - HFlip, VFlip 
 - Random rotate 90
 - Affine   
6. Threshold  
 Fixed threshold: 0.2 
7.  Ensemble  
 Weighted average: (1 * resnet34 + 1 * inceptionv3 + 1 * seresnet50 + 3 * seresnext50) / 6  

Cheers,

Edit: We use 3TTA: Flip, VFlip and Normal
