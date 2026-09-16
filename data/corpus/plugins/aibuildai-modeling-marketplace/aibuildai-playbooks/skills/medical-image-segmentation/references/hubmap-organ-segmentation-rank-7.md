# 7th place solution

Competition: hubmap-organ-segmentation
Rank: #7
Source: https://www.kaggle.com/c/hubmap-organ-segmentation/discussion/354859

Thanks to the competition host, Kaggle, and all participants. 
And congrats to all the winners!

# Overview
The key points of my solution are:
- Strong data augmentation
- Using whole and tile images simultaneously

# Model & Training
I used 9 CNN models.
- efficientnet b5 x 7
- convnext base x 2
    - 1-class segmentation : 7 models
    - Multi class segmentation : 2 models

##### Best single model training settings
- Encoder 
 - efficientnet b5 advprop pre-trained
- Decoder
 - UNet-based model that I modified
- Data augmentation
 - Resize
     - e.g.) prostate : `cv2.resize(img, dsize=None, fx= 0.4/6.263, fy= 0.4/6.263, interpolation=cv2.INTER_LINEAR)`
 - RandomCrop (p=0.5) 
     - If cropped : tile image
     - Else : whole image
 - ShiftScaleRotate
 - GaussNoise
 - GaussianBlur or MotionBlur
 - HorizontalFlip 
 - ColorJitter
 - RGBShift
- Other detailed training settings
 - Input image size : 800 x 800
 - 65 epochs
 - Loss: BCE loss + Tversky loss
 - Optimizer: RAdam
 - Using external data from GTEx Portal and HuBMAP - Hacking the Kidney
     - GTEx Portal : spleen, prostate, largeintestine
          - I only used 1 slide per organ because I couldn't find out that they helped my scores increased.
 - All images were used for training 
 - 1-class segmentation

##### Best single model scores
- Public HuBMAP : 0.596
- Private : 0.810


# Inference
##### TTA
- Whole image
    - Rotate (0, 90, 180, 270) 
    - HorizontalFlip + rotate (0, 90, 180, 270) 
- Tile Image
    - Crop size : 1900 x 1900
        - If image size < 1900 : no crop
        - If 1900 <= image size < 3000 : 4 crops (2 x 2 tiles)
        - If 3000 <= image size : 9 crops (3 x 3 tiles)

##### Ensemble
- Each model prediction : ((mean whole images) > threshold_1) + ((mean tile images) > threshold_1)
- Final prediction : Sum of each model prediction > threshold_2
    - lung threshold_2 : 0
    - kidney threshold_2 : 4
    - the others : 2

### Tips
My scores increased a little using this trick.
```
img = tifffile.imread("test_image.tiff").astype(np.float32)
img = np.clip(img+15, 0, 255).astype(np.uint8)
```

--------------------------------------------------------------------------

P.S.
I only used kaggle kernel for training :)
