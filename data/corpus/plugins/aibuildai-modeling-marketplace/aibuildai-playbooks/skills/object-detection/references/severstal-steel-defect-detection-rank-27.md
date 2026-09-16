# 27th place solution (8 classification and 8 segmentation models)

Competition: severstal-steel-defect-detection
Rank: #27
Source: https://www.kaggle.com/c/severstal-steel-defect-detection/discussion/115876

Better late than never, I would like to share a short write up of my solution.

My solution is just a combination of 16 models:

**Segmentation**
4 x PSNets with resnet152 backbones + 1 x FPN with resnet50 backbone, trained on full-sized images
3 models from the notebook of @lightforever

**Classification**
2 x efficientnet-b4 + 2 x efficientnet-b5, trained on 64 x 400 images
1 x resnext101_32x4d + 1 x densenet201, trained on 224 x 224 resized images
1 x resnet152 + 1 x resnet50, trained on full-sized images

**Training**
All my models were trained on 80% of random subsample of training data each using Catalyst and strong augmentations from Albumentations (it was my first serious experience with both libraries, so I have experimented with a lot of different augmentations, that seems to produce reasonable results):
```
A.VerticalFlip(p=0.5),
A.HorizontalFlip(p=0.5),
A.ShiftScaleRotate(rotate_limit=15,shift_limit = 0.01, p=0.3), A.GridDistortion(p=0.2)A.OpticalDistortion(p=0.2),
A.Blur(blur_limit = 1, p=0.2), 
A.CLAHE(clip_limit = 2, tile_grid_size = (16,16), p=0.2), 
A.HueSaturationValue(p=0.1), 
A.JpegCompression(quality_lower = 50, p=0.2), A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.2), A.RandomGamma(p=0.2)

+ A.RandomSizedCrop or A.Resize
```

For segmentation models I have used:
1) dice + pixelwise BCE loss if the class mask is not empty
2) pixelwise BCE + overall BCE (based on max predicted pixel value) if the class mask is empty

For classification models - BCE loss

**Inference**
I have made simultaneous inference for all models: I have read small batches of test images, applied different sets of transformations, and computed predictions for all models. That means, I have read every image only once and it made inference quite fast.

The prediction of different classification and segmentation models were averaged. More complex blending schemes worsened my scores.

**TTA**
Classification, TTA2: original image + horizontal or vertical or both flips (different for different models)
Segmentation, TTA2: original image + both horizontal and vertical flip simultaneously

**Postprocessing**
Classification thresholds [0.5, 0.5, 0.5, 0.5]
Segmentation-classification thresholds [0.5, 0.5, 0.5, 0.5]
The sum of both classification approaches thresholds [1.05, 1.05, 1.05, 1.05]
Segmentation thresholds [0.3, 0.3, 0.3, 0.3]
Min sizes [400, 400, 1400, 4000]

My second chosen solution had different thresholds for different classes, and better public LB score – but it turned out to be overfitted to public LB.
