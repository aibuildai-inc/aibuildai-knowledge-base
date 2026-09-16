# 5th place solution, poisson blending,detection and tracking

Competition: tensorflow-great-barrier-reef
Rank: #5
Source: https://www.kaggle.com/c/tensorflow-great-barrier-reef/discussion/308007

Congrats to all the winners, and thanks to organizers.

The following ideas and methods helped me survive from this challenging competition.
1.Copy COTS box and paste into background image, apply poisson blending.
2.Several detection models training and inference with different image sizes.
3.Appending boxes to detection results by finding homography matrix.
4.Trust Local CV.

## Data
As the dataset is relative small, if we have more samples we can benefit from them.
We can generate more star fishes and put them in different under-sea images.
[poisson.png]

The simplest way is to crop the star fishes in train-set and paste it to other images, but we can find two obvious problems.
1.The boundary and color is not real.
2.The COTS is not on reasonable places.

I solved these two problems by:

1.Poisson blending
2.Training a classification model to predict whether a box with COTS is real or fake.

This idea helped me to improve the score.

I did not stop my experiments, I tried to train GAN models to generate starfishes, although the generated samples were quite real, the score did not improve. 
I think it’s limited by the count of unique star fishes are small, if we can use more COTS downloaded from internet, I guess we could improve the score. I am not sure whether an image is allowed to use or not, so I did not use any of them. 
Other methods such as image harmonization did not bring  improvement.  
Anyway, despite quite a lot of time spent without too many gains in this competition, I quite enjoyed it.

## Model training

**Validation Strategy**
Split 5 folds by sequence

**Models**
YOLOv5-S6, YOLOv5-M6, YOLOv5-L6, YOLOX-L, YOLOR-P6 and HRNetV2P-W18

**Training details**
YOLOv5: 
Network: YOLOv5-S6, YOLOv5-M6, YOLOv5-L6
Training-size: 3600
Inference-Size: 4800
Optimizer: SGD
Scheduler: Warm Up + Linear LR + lr=0.01 + 15 epochs
Augmentation: hsv, translate, scale, flipud, fliplr, mosaic, mixup, water-augment, transpose

YOLOX:
Network: YOLOX-L
Training-size: 1280
Inference-Size: 1600
Optimizer: SGD
Scheduler: yoloxwarmcos + 20 epochs
Augmentation: hsv, flip, degrees, translate, shear, mosaic, mixup, no_aug_epochs=5

YOLOR:
Network: YOLOR-P6
Training-size: 2560
Inference-Size: 2560
Optimizer: SGD
Scheduler: Warm Up + Linear LR + lr=0.005 + 15 epochs
Augmentation: hsv, translate, scale, flipud, fliplr, mosaic, mixup, water-augment, transpose

HRNet:
Network: HRNetV2P-W18
Training-size: 3600
Inference-Size: 3600
Optimizer: SGD
Scheduler: Warm Up + linear LR + lr=0.02 + 10 epochs
Augmentation: Resize, RandomFlip

## Tracking
If the detection model has found a box in  previous frames and we can predict the box in the current frame by the following way:
1.Kalman Filter
2.Optic Flow
3.Finding homography matrix and then apply the matrix to get the box in the current frame.

I find the third method is the best for this dataset.
We can get keypoint descriptors by using SuperPoint/SuperGlue and then find homography matrix.

**The Tracking pipeline is:**
1.If there is a box detected a frame t-10, then append a box using the homography matrix to frame t-9 unless there already a box is there(IOU greater than 0.4). Then repeat this procedure from t-9 until current frame.
2.Apply DeepSort to get a track.
3.Determining a box on a track is kept or not by the ratio of the model predicted boxes. If the ratio is too low, the track may be False Positive.

## Ensemble
Calculating IOU between the boxes which predicted by different models in an image.
If max IOU with other boxes of a box is less than 0.55,then drop this box.
The remaining boxes are ensembled using WBF.


## Results
[result1.png]
[result2.png]
