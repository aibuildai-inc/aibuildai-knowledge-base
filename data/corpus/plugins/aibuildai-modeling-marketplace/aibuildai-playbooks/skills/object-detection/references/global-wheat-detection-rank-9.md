# [Update ] The 9th place solution with code

Competition: global-wheat-detection
Rank: #9
Source: https://www.kaggle.com/c/global-wheat-detection/discussion/172569

# Introduction
(This post was jointly written by the winning team, see details below)

First, we wish to express our congratulations to the winners and to all the nominees as well for their remarkable achievements and continued excellence and look forward to exciting times in object detection algorithms.

In addition, many thanks to Kaggle and the organizers for holding this interesting competition. And special thanks to my colleagues @orkatz2 (Or Katz) and @solomonk (Shlomo Kashani, author of the book “Deep Learning Interviews”).

# Code
- training code: https://github.com/amirassov/kaggle-global-wheat-detection
- submission code: [pseudo ensemble: detectors (3 st)+universenet r10](https://www.kaggle.com/amiras/pseudo-ensemble-detectors-3-st-universenet-r10)

# Summary
Our solution is based on the excellent MMDetection framework (https://github.com/open-mmlab/mmdetection).  We evaluated and tested numerous models in an attempt to find the ones which are better suited for wheat detection. Why? Because there is not always a direct correlation between the COCO mAP score, as reported by the authors of the original models, and the actual mAP on the wheat corpus. Therefore, experimentation was extremely important. Amongst the many models we evaluated in MMDet are: 
- GFocal
- ATSS
- UniverseNet 
- DetectorRS 
- SOLO-v2 (used for segmentation but we experimented with it anyway)

We then trained an ensemble of the top-performing models which were: 
- DetectoRS with the ResNet50 backbone (https://github.com/joe-siyuan-qiao/DetectoRS)
- UniverseNet+GFL with the Res2Net101 backbone (https://github.com/shinya7y/UniverseNet)

To increase the score a single round of pseudo labelling was applied to each model. Additionally, for a much better generalization of our models, we used heavy augmentations.

# Jigsaw puzzles
In the original corpus provided by the organizers, the training images were cropped from an original set of larger images. Therefore, we collected and assembled the original puzzles resulting in a corpus of 1330 puzzle images. The puzzle collection algorithm we adopted was based on https://github.com/lRomul/argus-tgs-salt/blob/master/mosaic/create_mosaic.py. But we were unsuccessful in collecting the bounding boxes for puzzles. Mainly because of the existence of bounding boxes that are located on or in the vicinity the border of the image. For this reason, we generated crops for the puzzles offline in addition to training images and generated boxes for them using pseudo labelling.

# Validation approach 
We used MultilabelStratifiedKFold with 5 folds (https://github.com/trent-b/iterative-stratification) stratified by the number of boxes, a median of box areas and source of images. We guaranteed that there isn’t any leak between the sub-folds, so that the images of one puzzle were used only in that one particular fold.

Referring to the paper https://arxiv.org/abs/2005.02162, one can see wheat heads from different sources. We assumed that the wheat heads of usask_1, ethz_1 sources are very different from the test sources (UTokyo_1, UTokyo_2, UQ_1, NAU_1). Therefore, we did not use these sources for validation.

However, our validation scores did not correlate well with the Kaggle LB. We only noticed global improvements (for example, DetectoRS is better than UniverseNet). Local improvements such as augmentation parameters, WBF parameters etc. did not correlate. We, therefore, shifted our attention to the LB scores mainly. 

We trained our models only on the first fold.

# Augmentations
Due to the relatively small size of our training set, and another test set distribution, our approach relied heavily on data augmentation. During training, we utilized an extensive data augmentation protocol:
- Various augmentations from albumentations library (https://albumentations.ai):
    - HorizontalFlip, ShiftScaleRotate, RandomRotate90
    - RandomBrightnessContrast, HueSaturationValue, RGBShift
    - RandomGamma
    - CLAHE
    - Blur, MotionBlur
    - GaussNoise
    - ImageCompression
    - CoarseDropout
- RandomBBoxesSafeCrop. Randomly select N boxes in the image and find their union. Then we cropped the image keeping this unified.
- Image colorization (https://www.kaggle.com/orkatz2/pytorch-pix-2-pix-for-image-colorization) 
- Style transfer (https://github.com/bethgelab/stylize-datasets). A random image from a small test (10 images) was used as a style.  
- Mosaic augmentation. `a, b, c, d` -- randomly selected images. Then we just do the following:
```
top = np.concatenate([a, b], axis=1)
bottom = np.concatenate([c, d], axis=1)
result = np.concatenate([top, bottom], axis=0)
```
- Mixup augmentation. `a, b` -- randomly selected images. Then: `result = (a + b) / 2`
- Multi-scale Training. In each iteration, the scale of image is randomly sampled from `[(768 + 32 * i, 768 + 32 * i) for i in range(25)]`.
- All augmentations except colorization and style transfer were applied online.
Examples of augmented images:

   |  
:-------------------------:|:-------------------------:
 | 

# External data:
SPIKE dataset: https://www.kaggle.com/c/global-wheat-detection/discussion/164346

# Models
We used DetectoRS with ResNet50 and UniverseNet+GFL with Res2Net101 as main models. DetectoRS was a little bit more accurate and however much slower to train than UniverseNet:
- Single DetectoRS Public LB score without pseudo labeling: 0.7592
- Single UniverseNet Public LB score without pseudo labeling: 0.7567

For DetectoRS we used:
- LabelSmoothCrossEntropyLoss with parameter 0.1
- Empirical Attention (https://github.com/open-mmlab/mmdetection/tree/master/configs/empirical_attention)

# Training pipeline
In general, we used a multi-stage training pipeline:


# Model inference
We used TTA6 (Test Time Augmentation) for all our models:
- Multi-scale Testing with scales [(1408, 1408), (1536, 1536)]
- Flips: [original, horizontal, vertical]

For TTA was used a standard MMDet algorithm with NMS that looks like this for two-stage detectors (DetectoRS):


For one-stage detectors (UniverseNet), the algorithm is similar, only without the part with RoiAlign, Head, etc.

# Pseudo labelling
- Sampling positive examples. We predicted the test image and received its scores and the bounding boxes. Then we calculated `confidence = np.mean(scores &gt; 0.75)`. If the confidence was greater than 0.6 we accepted this image and used for pseudo labelling.
- Sources [usask_1, ethz_1] and augmentations like mosaic, mixup, colorization, style transfer weren’t used for pseudo labelling.
- 1 epoch, 1 round, 1 stage.
- Data: original data + 3 x pseudo test data

# Ensemble
We used WBF (https://github.com/ZFTurbo/Weighted-Boxes-Fusion) for the ensemble. The distribution of DetectoRS and UniverseNet scores is different. So we applied scaling using `rankdata` (https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.rankdata.html):
```
scaled_scores = 0.5 * (rankdata(scores) / len(scores)) + 0.5.
```

WBF parameters:
- weights=[0.65, 0.35] respectively for models [DetectoRS, UniverseNet]
- iou_thr=0.55
- score_thr=0.45

# Our final submission pipeline (0.7262 on Private LB and 0.7734 on Public LB):


# Some observations from our submissions:
- Pseudo crops from jigsaw puzzles (DetectoRS R50): 0.7513 -&gt; 0.7582
- Tuning of pseudo labeling parameters for sampling positive examples (ensemble): 0.7709 -&gt; 0.7729
- Pseudo labeling (DetectoRS R50): 0.7582 -&gt; 0.7691
- Pseudo labeling (UniverseNet Res2Net50): 0.7494 -&gt; 0.7627
- SPIKE dataset (DetectoRS R50): 0.7582 -&gt; 0.7592
- Deleting [usask1, ethz1] from pseudo labeling (DetectoRS R50): 0.7678 -&gt; 0.7691

# What we didn’t do:
- MMDetection and YOLOV5 ensemble. That was the main goal but since YOLOV5 was disqualified we gave up this option.
- MMDetection and EfficientDet ensemble. We only managed to make a submission for the first time on the last day but we could not get a good enough result. We believe that if we had a few more days we could improve the result by tune the WBF parameters.
- We stopped improving YOLOV5 about a month before the end of the competition. 

# What didn’t work:
- Wheat Ears Detection Dataset: CV improved, but LB did not.
- More than 1 rounds of pseudo labeling with the same model.
- Scale-aware testing like this: https://drive.google.com/file/d/14VwSjMeRZUtisZtqQPmbll6w4zvZXIAQ
- Rotate90 TTA for DetectoRS and UniverseNet

# Acknowledgements
Thanks to internet service provider HOSTKEY (https://www.hostkey.com) that gave me a grant for access to GPU servers.


**The code will be published soon.**
