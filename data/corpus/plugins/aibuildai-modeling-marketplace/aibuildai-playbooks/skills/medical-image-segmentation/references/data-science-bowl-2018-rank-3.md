# 3th place on private LB: Matterport's Mask-RCNN

Competition: data-science-bowl-2018
Rank: #3
Source: https://www.kaggle.com/c/data-science-bowl-2018/discussion/56393

The third place solution, tie with #2 jacobkie achieving 0.614 on the Private Leader-board, is based on a single  Mask-RCNN model using as code-base Matterport's Mask-RCNN (https://github.com/matterport/Mask_RCNN). 

**Summary**


----------


I don't think I have done many different things to what others have reported and my solution is quite simple. I'll be happy to share the code once I have the approval by kaggle and sponsors. I tried many different things, but the main two contributions are:

1) **Strong scaling augmentation**, a lot of zooming in and out and aspect ratio changes before taking the 512x512 crops used as inputs to the model during training.

2) **Test time augmentation**, I used 15 different augmentations at test time with different rotations, scalings, channel color shifts, etc. This takes a loooong time (aprox. 2 days for the stage_2 test set) and a binary dilation post-processing actually gives a very similar score, so I would use the latter if asked now (although it is easy to tell now that we can see the PL scores..)


**Training data**


----------


No external data was used, only stage 1 training set. I used the corrected data-set compiled in: https://github.com/lopuhin/kaggle-dsbowl-2018-dataset-fixes 
I didn't have the time to include any of the external data, I was also afraid that the different annotation styles might introduced unwanted bias in the predictions.
No prepocessing.

**Augmentations**


----------


In addition to the scaling augmentation mentioned above I used left-right and up-down flips, random 90 degree rotations, random additional rotation on top of those, random channel color shifts 

**Parameters**


----------


Here are some of the parameters for comparison (https://www.kaggle.com/c/data-science-bowl-2018/discussion/54920):

CodeBase	Type-1 and 2
MEAN_PIXEL	[123.7, 116,8, 103,9]
LEARNING_RATE	Start 0.001 and down to 3*10^-5
LEARNING_SCHEDULE	~120 always "all"
RPN_ANCHOR_RATIOS	[0.5, 1, 2]
USE_MINI_MASK	True
MINI_MASK_SHAPE	(56,56)
GPU_COUNT	1
IMAGES_PER_GPU	2
STEPS_PER_EPOCH	332
VALIDATION_STEPS	0
BACKBONE	resnet101
NUM_CLASSES	1+1
IMAGE_MIN_DIM	512
IMAGE_MAX_DIM	Not used
IMAGE_PADDING	Not used
RPN_ANCHOR_SCALES	8,16,32,64,128
RPN_ANCHOR_STRIDE	1
BACKBONE_STRIDES	4,8,16,32,64
RPN_TRAIN_ANCHORS_PER_IMAGE	256
IMAGE_MIN_SCALE	Not used
IMAGE_RESIZE_MODE	crop at training, pad64 for inference
RPN_NMS_THRESHOLD	0.7
DETECTION_MIN_CONFIDENCE	0.9
DETECTION_NMS_THRESHOLD	0.2
TRAIN_ROIS_PER_IMAGE	600
DETECTION_MAX_INSTANCES	512
MAX_GT_INSTANCES	256
init_with	coco
DATA_AUGMENTATION	scaling, crop, flip-lr, flip-up, 90 rotation, rotation, channel_shift

----------
Source code is now available [follow this link to github][1]



  [1]: https://github.com/Gelu74/DSB_2018
