# 9th place solution

Competition: hubmap-hacking-the-human-vasculature
Rank: #9
Source: https://www.kaggle.com/c/hubmap-hacking-the-human-vasculature/discussion/428447

Thanks to the organizers and congrats to all the winners!

## Overview
From the top 1&2 solutions of the [Sartorius competition](https://www.kaggle.com/competitions/sartorius-cell-instance-segmentation), I decided to go with a two-stage pipeline of object detection and semantic segmentation instead of using a one-stage model (e.g. Mask-RCNN) in the early in the competition. I think that the main advantage of the two-stage pipeline is the ease of the ensemble and TTA.



## Detection part

### Update of dataset2 bbox-level annotation 
As been pointed out in the discussion, the dilation significantly improved LB score in models trained from both dataset1 and dataset2. However, models trained only from dataset1 did not show this effect. This led me to believe that the annotations of dataset2 were made smaller than those of dataset1. Considering that there are several types of blood vessels and the possibility that the annotations of them are not uniformly small, I took the approach of updating the bbox-level annotations of dataset2 to dilate them by models trained only from dataset1.



The update was performed by replacing the original annotations with a predicted ones that met the following conditions. Basically, the updated bboxes is larger than the original ones. The total number of bboxes in dataset2 does not change by the updating.
1. IoU > 0.4
2. FP / (TP + FP) > 0.1



In the model trained from the updated dataset2 together with dataset1, the LB improvement by the dilation has almost disappeared and the LB score above 0.5 was achieved without the dilation.

### CV strategy
The CV was carried out by the following special two-fold division.



### Model training
I trained 2-class (blood_vessel, glomerulus) detection models. "unsure" label was ignored. I modified the source code of YOLOv5/v7/v8 and add a 90 degree random rotation augmentation. The following 5 models x 2 folds (total 10 models) were used in the final submission.

| Model | Input size |
| ---- | ---- |
| YOLOv5x6 | 512 |
| YOLOv7x | 512 |
| YOLOv8l | 512 |
| YOLOv8x | 512 |
| YOLOv8l | 768 |

### Inference
I made a huge ensemble of 10 models x 16 TTAs since I considered the accuracy of detection to be more important than one of the segmentation. The small number of test images made it possible. 10 x 16 = 160 detection results were merged by WBF.

- 16 TTAs: 8 for combinations of h-flip, v-flip, 90deg rotation, 2 for 2 scales (base size, base size + 64px), 8 x 2 = 16
- IoU threshold of each model's NMS: 0.6
- IoU threshold of WBF: 0.7


## Segmentation part
### CV strategy
Almost the same as detection models, except that dataset2 is not updated.

### Model training
As a mask of bboxes, I did not use the prediction results of the detection models, but used the bboxes obtained from the original annotations. Unlike the detection models, "unsure" label was also used for training. In the final submission, EfficientNetB1-Unet and EfficientNetB2-Unet was used (each 2 folds, total 4 models).

### Inference
TTA was not used because of run-time constraints. A score threshold of 0.5 was used for binarization.

## Strategy of final submission
The effect was smaller by updating dataset2, but the small dilation improved the LB score slightly. I implemented the dilation not by cv2.dilate for the final masks, but by increasing the size of bboxes by a percentage. In my final submission, 3% of bbox dilation increased my LB score about 0.005. I used 3% dilation in one of the two final submissions and not in the other (there are other differences besides the dilation).

| Submission | Public LB | Private LB |
| ---- | ---- | ---- |
| w/ 3% dilation | 0.580 | 0.549 |
| w/o 3% dilation | 0.572 | 0.560 |


## Things tried but not worked
- Semi-supervised learning on dataset3
