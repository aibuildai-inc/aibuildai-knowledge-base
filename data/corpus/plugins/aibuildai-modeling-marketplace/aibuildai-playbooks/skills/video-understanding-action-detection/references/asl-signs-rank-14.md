# 14th place solution: publicly shared Transformer architecture was so strong!

Competition: asl-signs
Rank: #14
Source: https://www.kaggle.com/c/asl-signs/discussion/406301

# Summary

I thank Kaggle administraror & host for holding this competition. Although struggling with Tensorflow's unfriendly errors and lots of try-and-errors for failing TFLite conversion was really, really tough, these low-layer experience was valuable for me.
Below is my solution writeup of this competition.

## Pipeline

I tried over 377 different patterns of training models for this competition, however, the best architecture is only minor-changed one from [Mark Wijkhuizen's great public notebook](https://www.kaggle.com/code/markwijkhuizen/gislr-tf-data-processing-transformer-training).



### Mixed PostLN & PreLN Architecture

I tested a) PostLN, b) PreLN, c) Mixed architectures, and found Mixed architecture provides the best result. This architecture was originally (perhaps unintentionally) implemented in the Mark Wijkhuizen's public notebook (in earlier version).



### Other modifications

* keep frames with no hands (instead of dropping) in pre-processing.
* increase number of layers in the keypoint encoder. With more layers, the accuracy gets better. The 4x layer seetting is the best tradeoff for accuracy and inference time.
* set number of hidden units in keypoint encoder independently for each parts (lips=192, left_hand=256, right_hand=256, pose=128). This reduces inference time without losing accuracy.
* attach ArcFace layer on training.

## Training Setting

* loss function: `0.5 * ArcFace + 0.5 * CrossEntropy` (This setting was shared by [Med Ali Bouchhioua](https://www.kaggle.com/code/medali1992/gislr-nn-arcface-baseline)). Using ArcFace loss together with cross entropy loss converges faster, as well as the accuracy gets better.
* I tested 50, 80, 100, 120 epochs, but 100 epoch is the best on LB.
* Label smoothing (0.20-0.25) can avoid overfitting, but ArcFace is better. Using both label smoothing and ArcFace didn't increase CV/LB.

### Data Augmentation

The augmentation strategy is almost the same as [that of @hengck23 was shared](https://www.kaggle.com/competitions/asl-signs/discussion/391265).

* handedness swapping (p=0.5)
* global 2D Affine transformation (p=0.5, shift=(-0.1, 0.1), rotation=(-30, 30), scale=(0.9, 1.1), shear=(-1.5, 1.5))
* random frame masking (p=0.5, mask_ratio=0.75).

Frame masking simulates the mis-detection of keypoins. It also act as cutout augmentation in image tasks.

## TFLite Conversion

When FP16 quantization is applied for the Mark's original implementation, the Transformer block outputs NaN value. So I rewrote transformer block based on [@henck23's implementation](https://www.kaggle.com/code/hengck23/lb-0-73-single-fold-transformer-architecture/notebook).
FP16 quantization made model size about 1/2, without losing model accuracy.

### TTA

* apply handedness swapping augmentation to 2/4 of ensemble-seed models

### The output TFLite model

* model size=26MB
* scoring time=56-59 min

## Not-worked Experiments

* distance/velocity features didn't contribute to increase accuracy.
* using synthesized hand pose dataset, I trained z-axis prediction model. I used this model's prediction result for a) 3D-affine transform augmentation and b) sub-task to predict z-axis from key-point encoder's output, but both trials didn't contribute to increase accuracy.
