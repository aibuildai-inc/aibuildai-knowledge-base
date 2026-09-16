# 12th place solution - YOLOv5 + Optical flow tracker

Competition: tensorflow-great-barrier-reef
Rank: #12
Source: https://www.kaggle.com/c/tensorflow-great-barrier-reef/discussion/307753

First of all, I would like to thank the host for organizing such an exciting competition. I am glad to survive this big shake!

## Detection model training
Because I utilized only the kaggle-kernel's GPU, I wanted to reduce GPU memory usage and training time. So I upscaled the training images by two times (2560x1440) and split them into four 1280x720 images. After that, I discarded the split image not containing bounding boxes. So that means the training size was 1280. In the inference phase, 2560 size was used. By doing this, I thought that the results are almost same to those obtained when training with 2560 size with reduced memory usage.
 
- YOLOv5m6
- 4 folds Group-Kfold grouped by sequence
- 1280 size with x2 upsacaled & 4 split images
- Augmentation: Mainly, flipud p=0.5, mixup p=0.5, rot90 p=0.5 was changed from "hyp.scratch.yaml". To implement rot90, I modified "augmentations.py".

## Tracking
I reused my optical flow based tracking code which I made in the [NFL competition](https://www.kaggle.com/c/nfl-health-and-safety-helmet-assignment/discussion/285156#1569583). Utilizing optical flow, next frame's bouding boxes can be estimated from previous frame's detection result. So it can continue tracking even if the detection model lost detection. I varied the number of frames continuing tracking in the lost detection condition according to the number of tracking counts up to that frame, and its maximum number is five.


## Choice of final submission
In my CV, I noticed that upscaling the image size equally both of training size and inference size from original 1280x720 size contributed to both CV score and LB score. However, using different  size between training and inference (for example training 2560 and inference 3840) was only contributed to LB score and got worse the CV score. So I suspected the possibility of overfitting and I chose the following four submissions.

##### 1. Best public LB
- Ensemble of following 3 models
    - Best 2 public LB score models out of 4 folds with inference size of 3840 (x1.5 upscaled from training size)
    - YOLOv5s model shared by @freshair1996 (score 0.665) with inference size of 6400
- TTA: original, LR flip, UD flip (*)
- Ensemble: 3 models x 3 TTAs were ensembled by WBF
(*) The TTA pattern was limited by 9 hours limitation. If more TTA pattern is used, the more score will be got.

##### 2. Best CV
- All of 4 folds with inference size of 2560 (= training size)
- TTA: original, LR flip, UD flip, Rot90
- Ensemble: 4 models x 4 TTAs were ensembled by WBF

##### 3. Middle of CV and public LB
- Best 2 public LB score models out of 4 folds
- TTA: 2 inference size (2560, 3840) x original, LR flip, UD flip
- Ensemble: 2 models x 2 sizes x 3 TTAs were ensembled by WBF

##### 4. TensorFlow EfficientDet
To get the TensorFlow prize, I trained the EfficientDet-D2 model, but the private LB score of it was 0.543...

The scores are following.

| submission | Private LB w/ Tracking | Private LB w/o Tracking | Public LB w/ Tracking | Public LB w/o Tracking |
| --- | --- | --- | --- | --- |
| 1 | 0.712 | 0.697 | **0.718** | 0.703 |
| 2 | 0.708 | 0.700 | 0.621 | 0.609 |
| 3 | **0.718** | 0.705 | 0.667 | 0.658 |

\# Updated 13th place to 12th place
