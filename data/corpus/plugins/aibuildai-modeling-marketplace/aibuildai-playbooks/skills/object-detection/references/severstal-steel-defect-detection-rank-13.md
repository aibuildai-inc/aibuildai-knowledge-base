# 13th place solution

Competition: severstal-steel-defect-detection
Rank: #13
Source: https://www.kaggle.com/c/severstal-steel-defect-detection/discussion/114332

First of all, thanks to Severstal for hosting this competition. And congrats to the winners!

I was very surprised that such a huge shake-up and I am very fortunate to get a solo gold medal. I have tried on this competition only about 10 days, so I haven't done many experiments and my solution is not so special. I was just very lucky.

In a short words, my solution is just an ensemble multiple models.  From my experience so far, I think it is very important to ensemble various models to build a robust solution.

### Classification
* models
  * 3x EfficientNet-b4 (first 3 of stratified 10 folds)
* input
  * full size (256 x 1600)
* augmentations
  * random crop rescale
  * hflip, vflip
  * random contrast, random gamma, random brightness
* TTA
  * none, hflip
* threshold label
  * [0.5, 0.5, 0.5, 0.5]

### Segmentation
* models
  * EfficientNet-b3 Unet stratified 4fold w/ full size image
  * EfficientNet-b3 Unet 1fold w/ random crop 256 x 800
  * 3x Unet from [mlcomp + catalyst infer kernel](https://www.kaggle.com/lightforever/severstal-mlcomp-catalyst-infer-0-90672) from @lightforever 
* augmentations
  * same as classification
* loss function
  * BCEDice (bce weiht=0.75, dice weight=0.25)
* TTA
  * none, hflip
* threshold mask
  * [0.5, 0.5, 0.5, 0.5]
* postprocess

### Predictions on public LB
defect 1 = 82 (128)
defect 2 = 5 (43)
defect 3 = 601 (741)
defect 4 = 110 (120)
