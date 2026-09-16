# 3rd Place Solution

Competition: czii-cryo-et-object-identification
Rank: #3
Source: https://www.kaggle.com/c/czii-cryo-et-object-identification/discussion/561417

Thanks kaggle&host for this interesting competition. 
What I like for this competition is that the host give out a baseline(especially for data processing), it is very helpful for people like me who has no knowledge with this domain.
Another reason I join this competition was that I want to writing training code based on the library accelerate(I use pure pytorch before). 
Since I am on the top, I will think there is no critical bug in my training code.


# Summary
Before I start this competition I thought it was a OD task not a segmentation task untile @hengck23 publish his good [notebook](https://www.kaggle.com/code/hengck23/3d-unet-using-2d-image-encoder).
My solution is based on 3D unet with post processing using cc3d. Cross Entropy loss is used with all the 7 particles.
My best solution is 4 fold(7 KF) average ensemble of backbone res101.

One thing I want to mention is that my best solution has same score(0.783) on both Public and Private LB.

# models
I use the code from [segmentation_models_pytorch_3d](https://github.com/ZFTurbo/segmentation_models_pytorch_3d).
The best solution is Unet + resnet101.
I also try different architecture with different backbones, but unet+resnet101 is the best on public LB.
I did not dig much of the code, so mostly default parameters was used for these models.

# train

I use EMA because it is easier to handle than SWA, although I remember there is saying SWA is better than EMA.
I train the model with input size (64, 128, 128), but inference with (64, 256, 256), which give 0.001 improvement on public LB.
I use half of the original radius during training, considering how the evaluation score is calculated, which is also best based on my experiment.

# Augmentation
It is obvious data augmentation will help a lot for this competition considering the rare data we have.
What I used:
- flip on axis x, y, z
- switch axis x and y
- different algos: "denoised", "wbp", "ctfdeconvolved", "isonetcorrected
- simple copy past
- mixup

# TTA
2 tta was used, output is averaged with original:
- flip x, y, z
- rot90 for x, y
  
# ensemble
- 4 fold of 7KF average ensemble

# Failures
- Try to pretrain on the external data provided by host
- ensemble unet with different backbone like resnet34 and resnet10
