# 5th Place Solution (EfficientDet)

Competition: global-wheat-detection
Rank: #5
Source: https://www.kaggle.com/c/global-wheat-detection/discussion/172458

Thank you for host and competitors.
It is interesting competition for me.

# TL;DR
1. EfficientDetB3 5Fold -&gt; 10 image Test with Pseudo Labeling
2. EfficientDetB5 All Training(EMA) -&gt; Prediction MultiScaleTraning(1280,768,1024+4Flip) 0.766
3. EfficientDetB5 All + EfficientDetB4 All + EfficientDetB5 All(With Pseudo) Public 0.7752

# Solution
## 1st Phase
EfficientDetB3 for test 10 images pseudo labeling
Test image is "test source", I want to get test source knowledge

## 2nd Phase
I used EMA for the ensemble technique in 2nd phase.

1. AdamW 100epoch 640 batchsize 4(for bn parameter turning)
2. AdamW 40epoch 1024 batchsize 1(bn layer freeze)

use Cosine Annealing

Augmentation
- Mixup
- Mosaic
- Scale
- Hue
- Random Brightness
- Cutout
- GridMask

## 3rd Phase(Kernel)
### Prediction
EfficientDetB5 All + Predict MultiScale(768, 1024, 1280 * 4Flip) -&gt; Pseudo Labeling of All test. 

### Pseudo Labeling
Traning Parameter as follows
- EfficientDet B5 image size 1024  
- Epoch 5
- Use EMA
- Mixed Precision(AMP)

### Ensemble
EfficientDetB5 All + EfficientDetB4 All + EfficientDetB5 All(With Pseudo) using WBF

# Why Shakedown?
1. failed Threshold turning.(lower threshold is better in private)  
   When setting threshold 0.1, the score go to 0.695. but I can not get selection.
2. Use more heavy augmentation for robustness
