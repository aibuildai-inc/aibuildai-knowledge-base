# 1st place solution [MIT-Compliant]

Competition: global-wheat-detection
Rank: #1
Source: https://www.kaggle.com/c/global-wheat-detection/discussion/172418

Thanks to the organizers for this interesting competition!
Congratulation to all, this is my first solo gold medal. As there are lots of great public notebooks posted, I will keep my solution simple and short.
Special thank to @rwightman for amazing repo (https://github.com/rwightman/efficientdet-pytorch)

**Summary**
- Custom mosaic data augmentation
- MixUp
- Heavy augmentation
- Data cleaning
- EfficientDet
- Faster RCNN FPN
- Ensemble multi-scale model: Weighted-Boxes-Fusion, special thank @zfturbo 
- Test time augmentation(HorizontalFlip, VerticalFlip, Rotate90)
- Pseudo labeling

**Data Processing**
- Custom mosaic augmentation:
Mosaic is a data augmentation method that combines 4 training images into one for training (instead of 2 in CutMix). Instead of randomly cropping a piece of the image, I create custom mosaic augmentation as bellow to keep the border information:


- MixUp: 


- Heavy augmentation:
RandomCrop, HorizontalFlip, VerticalFlip, ToGray, IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, Blur, CLAHE, Sharpen, Emboss, RandomBrightnessContrast, HueSaturationValue

examples after apply mosaic, mixup, augmentation


- External data
I use 2 external data:
*wheat spikes (https://www.kaggle.com/c/global-wheat-detection/discussion/164346), for license please refer (https://www.kaggle.com/c/global-wheat-detection/discussion/164346#928613)
*wheat 2017 (https://plantimages.nottingham.ac.uk/) at post (https://www.kaggle.com/c/global-wheat-detection/discussion/148561#863159). 
I create annotations (bounding box) for all images and crop to 1024x1024. I contacted author to ensure the right without license issue.
- Data Cleaning
*Deleted tiny bounding boxes (width or height < 10px)
*Fixed too big bounding boxes

**Model**
- 5 folds, stratified-kfold, splitted by source(usask_1, arvalis_1, arvalis_2...)
- Optimizer: Adam with initial LR 5e-4 for EfficientDet and SGD with initial LR 5e-3 for Faster RCNN FPN
- LR scheduler: cosine-annealing
- Mixed precision training with nvidia-apex

**Performance**
Valid AP/Public LB AP
- EfficientDet-d7 image-size 768: Fold0 0.709/0.746, Fold1 0.716/0.750, Fold 2 0.707/0.749, Fold3 0.716/0.748, Fold4 0.713/0.740
- EfficientDet-d7 image-size 1024: Fold1,3
- EfficientDet-d5 image-size 512: Fold4
- Faster RCNN FPN-resnet152 image-size 1024: Fold1
- Ensemble 9 models above using wbf can achieve **0.7629 Public LB/0.7096 Private LB** (old testset)

**Pseudo labeling**
- Base: EfficientDet-d6 image-size 640 Fold1 0.716 Valid AP
- Round1: Train EfficientDet-d6 10 epochs with trainset + hidden testset (output of ensembling), load weight from base checkpoint
  Result: [old testset] 0.7719 Public LB/0.7175 Private LB and [new testset] 0.7633 Public LB/0.6787 Private LB
- Round2: Continue train EfficientDet-d6 6 epochs with trainset + hidden testset (output of pseudo labeling round1), load weight from pseudo labeling round1 checkpoint
  Result: [old testset]0.7754 Public LB/0.7205 Private LB and [new testset]0.7656 Public LB/0.6897 Private LB

Github source code at https://github.com/dungnb1333/global-wheat-dection-2020
