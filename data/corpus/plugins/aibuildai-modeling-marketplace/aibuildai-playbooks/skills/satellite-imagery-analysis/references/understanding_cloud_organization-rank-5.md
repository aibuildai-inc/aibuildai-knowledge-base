# 5th place solution(single segmentation model private lb 0.66806)

Competition: understanding_cloud_organization
Rank: #5
Source: https://www.kaggle.com/c/understanding_cloud_organization/discussion/117987

Congratulations to all winners in this competition! This is my first gold medal. I feel so happy.

My solution is ensemble of 3 segmentation models
## Augmentation
In this competition I found image augmentation is very important. I have tried many different augmentation sets and finally found one set that works good for me. I use albumentation to do image augmentation.
```
aug = Compose([
        ShiftScaleRotate(scale_limit=0.5, rotate_limit=0, shift_limit=0.1, p=0.6, border_mode=0),
        OneOf([
            ElasticTransform(p=0.5, alpha=50, sigma=120 * 0.02, alpha_affine=120 * 0.02),
            GridDistortion(p=0.5),
            OpticalDistortion(p=0.5, distort_limit=0.4, shift_limit=0.5)
        ], p=0.8),
        RandomRotate90(p=0.5),
        Resize(352, 544),
        VerticalFlip(p=0.5),
        HorizontalFlip(p=0.5),
        OneOf([
            IAASharpen(alpha=(0.1, 0.3), p=0.5),
            CLAHE(p=0.8),
            GaussNoise(var_limit=(10.0, 50.0), p=0.5),
            #GaussianBlur(blur_limit=3, p=0.5),
            ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=0.3),
        ], p=0.8),
        RandomBrightnessContrast(p=0.8),
        RandomGamma(p=0.8)])
```
## Models
**Model1: **
```
Encoder: efficientnet-b1
Decoder: unet
Image Input Size: 416x608
TTA: hflip, vflip, multi-scale: [(352, 544), (384, 576), (448, 640), (480, 672)] 
Threshold: threshold label = [0.85, 0.92, 0.85, 0.85], threshold pixel = [0.21, 0.44, 0.4, 0.3]
Score: 9-fold cv = 0.66002, public lb = 0.67070, private lb = 0.66806
```
**Model2:**
```
Encoder: efficientnet-b3
Decoder: fpn
Image Input Size: 352x544
TTA: hflip, vflip, multi-scale: [(320, 512), (384, 576)]
Threshold: threshold label = [0.85, 0.9, 0.9, 0.85], threshold pixel = [0.35, 0.4, 0.42, 0.42]
Score: 9-fold cv = 0.65646, public lb = 0.66426, private lb = 0.66687
```
**Model3:**
```
Encoder: resnet50
Decoder: unet
Image Input Size: 352x544
TTA: hflip, vflip, multi-scale: [(320, 512), (384, 576)]
Threshold: threshold label = [0.9, 0.92, 0.87, 0.82], threshold pixel = [0.35, 0.51, 0.31, 0.3]
Score: 9-fold cv = 0.65715, public lb = 0.66541, private lb = 0.65973
```

All models use bcedice loss and Adam optimizer. Run threshold search to get the threshold label and threshold pixel
## Ensemble
I use cv and public lb score to roughly set model weights, and run threshold search to get the threshold.
```
Model Weight: model1, model2, model3 = [4, 1, 2]
Threshold: threshold label = [0.84, 0.9, 0.85, 0.8], threshold pixel = [0.25, 0.43, 0.35, 0.35]
Score: 9-fold cv = 0.66449, public lb = 0.67601, private lb = 0.67080
```

Finally thanks to  @hengck23 rKeng,  I learn a lot from his code and ideas.
