# 22th place solution - Train4Ever

Competition: tensorflow-great-barrier-reef
Rank: #22
Source: https://www.kaggle.com/c/tensorflow-great-barrier-reef/discussion/307622

Thanks to the hosts for providing an interesting competition. Although not ending up in Gold zone, we would like to share our solution. 

### Our solution consists of 3 stages:
- Detection
- Tracking
- Classification post processing

### 1. Detection
- Using some popular models like everyone does. We used CascadeRCNN ResNeSt200, YoloV5, YoloX, YoloR. 
- Ensembling: WBF

### 2. Tracking:
- We did not invest enough time to find a better tracking scheme than the one from the public notebook. As a result, we used that Norfair without any significant modification. 
- After Norfair is applied, it is certain that the predictions contain so many FPs though it does increase the recall. It is the work of stage 3 that reduce the effect of the problem. 

### 3. Classification:
- This is important in our pipeline because we do not have too strong detectors and tracking method.
- Flow: detectors predict bboxes -> crop -> classification -> blending
- We have 5 folds. In each fold, each of the 4 detector makes prediction on both train and validation set. We focus on hard false positive (high detection score but wrong) and hard false negative (low detection score but true).  The predictions on train set from all the detectors are concatenated and reduce overlapping by NMS, then used to train the classification models. The same method is used to produce the validation set for the classification models.
- We also crawled some COTS data from the Internet with Creative Commons license to generalize the model more.
- Models: EfficientnetB7, Eca Nfnet L0
- Im size: 128
- Augmentations: heavy augmentations. Cutmix only when training EffB7 
- Training notebooks: https://drive.google.com/file/d/1uD4Wj4pxfScm-NnXuf9jSVXCru-XTGJB/view?usp=sharing (Eca nfnet l0) , https://drive.google.com/file/d/1pQuk6z5Vl7sfg7v36NVjk3WpbYYglS91/view?usp=sharing (EffB7)
- 2 model type x 5 folds = 10 models classification for final submission
- Blending method: Use geometric mean between the classification probability score and the bbox conf score from the detectors.
- We estimated this boosted about 6% on the private LB in our final 0.706 solution.
