# 19th place solution with code

Competition: understanding_cloud_organization
Rank: #19
Source: https://www.kaggle.com/c/understanding_cloud_organization/discussion/117975

Congrats for all the prize winner and who finished in gold zone!

I joined this competition relatively lately, after Severstal competition finished.(I believe same as many people, don't you?) 
My purpose was to make sure the segmentation pipeline I made in Severstal works for other competition. And it turned out it actually works, I just modified directory and some small parameters. That means my solution is not so special, honestly.
<br>
### Overview
- Extremely noisy annotation
- Not so imbalanced classes (compared with Severstal)
- Relatively small data(number of samples)
- Good train/test split(cv works)
<br>
### What works
- Unet &amp; FPN
- not so large encoder
- BCE + Dice loss
- heavy augmentation(including mixup)
- cosine anealing
- ensemble many models
- Triplet thresholding(label threshold/mask threshold/min componet)
<br>
### What didn't works
- PSPNet
- large image size(over 448*672)
- plane BCE
- pseudo labeling

### Solution
1. Unet/efficientnet-b3/image size 320x480/5fold
2. Unet/efficientnet-b0/image size 320x480/cosineanealing/5fold
3. Unet/efficientnet-b3/image size 384x576/cosineanealing/5fold
4. FPN/resnet34/image size 384x576/mixup/5fold
5. Ensemble above 20 models
6. Triplet thresholding(label threshold/mask threshold/min componet)

----
Here is my code.  
If you have question, please feel free to ask:)
Thanks!

https://github.com/bamps53/kaggle-cloud-2019
