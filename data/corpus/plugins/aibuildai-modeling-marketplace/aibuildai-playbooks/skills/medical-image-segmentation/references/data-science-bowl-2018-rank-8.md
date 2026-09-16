# My Solution 11th place (LB 0.590)

Competition: data-science-bowl-2018
Rank: #8
Source: https://www.kaggle.com/c/data-science-bowl-2018/discussion/54838

（currently 8th place while leaderboard is being updated）

Hi all. It is my first time join this competition. All my methods are learned from open discussion. As return, I decide to share my solution to the community. Thanks a lot to those who have made many contributions to the community. 

First of all, I thank Allen, John1231983, Keven Wang, Mehul Sampat, Heng CherKeng, CPMP, bsp2020, kamil, YaGana Sheriff-Hussaini, Yan Wang, Konstantin Maksimov, xiapidan, xuan, Waleed for your very impressive discussions. I learned a lot from those discussions. @Mehul Sampat hope we can team up together next time :) thanks again.

My score: Stage1 LB 0.489(87th), Stage2 LB 0.590(11th). I am using matterport mask-rcnn.

**Augmentations**

Since there are hundreds of training images, we must find some useful augmentations to prevent our models from overfitting and make them generalizable. Here are some methods I have tried but it didn't work for me:
 
 - add gaussian noise
 - color to gray
 - contrast and brightness
 - random crop 512x512 if image size is bigger than 512 otherwise resize the image to 512x512
 - mosacis
 - mosacis+random crop
 - mosacis+random crop+ h&amp;e
 - rotate 90 degrees 
 - radnom rotate 90,180,270 degrees
 - rotate +-5 degrees on top of flip &amp; 90 degree rotation
 - elastic transform

I only use flip up&amp;down&amp;left&amp;right

**Additional data**

No. I have tried to add h&amp;e dataset to training data, but it didn't improve my performance. I am using https://github.com/lopuhin/kaggle-dsbowl-2018-dataset-fixes

**Ensembling**

No. I spend almost one week trying this method but it didn't work so well. I devided the training data into two categories: color &amp; grey. I have seen someone get a high score (0.5+) in stage1 by using this method. I am very interested in this so I really hope that someone can share some solutions about this.

**Parameters**

train:

inti_with= coco

RESNET_ARCHITECTURE = "resnet101"

MEAN_PIXEL = np.array([0., 0., 0.])

RPN_NMS_THRESHOLD = 0.7

DETECTION_MIN_CONFIDENCE = 0.7

DETECTION_NMS_THRESHOLD = 0.3

TRAIN_ROIS_PER_IMAGE = 600

RPN_TRAIN_ANCHORS_PER_IMAGE = 320

LEARNING_RATE = 1e-3

inference:

RPN_NMS_THRESHOLD = 0.6

DETECTION_NMS_THRESHOLD = 0.1

**Training**

1e-3 all 20epochs. Choose the best point according to the display of tensorboard, then 1e-4 or 1e-5 train all 20epochs. (if val loss stops decreasing, stops training) 
optimizer=Adam.

**Post processing** 
Use binary_dilation.

**Other method**
general standardization.
img = img-mean(img)/std(img).

**Some methods might be useful**

1.Filter

&gt; We use single mask rcnn, but with special post process to filter noises (i.e. small FPs) based on clustering and outlier detection of mask sizes, which boosted us a lot (kill lots of noises). Less FPs is the key to get high mAP as Heng said. I should thank him a lot

2.fill_holes

https://docs.scipy.org/doc/scipy-0.15.1/reference/generated/scipy.ndimage.morphology.binary_fill_holes.html#scipy-ndimage-morphology-binary-fill-holes

3.wateshed

**Some interesting discussions&amp;methods links**

 1. https://github.com/matterport/Mask_RCNN/issues/230
 2. https://github.com/matterport/Mask_RCNN/issues/281
 3. https://github.com/killthekitten/kaggle-ds-bowl-2018-baseline/issues/5
 4. https://www.kaggle.com/bostjanm/overlapping-objects-separation-method/notebook
 5. https://www.kaggle.com/c/data-science-bowl-2018/discussion/52989#30758

**Last**

I have tried a lot of experiments. Some of them i just don't remeber.....sorry :(  So here is my email mdlszhengli@@gmail.com. I am very happy to discuss via email or wechat(removed).
