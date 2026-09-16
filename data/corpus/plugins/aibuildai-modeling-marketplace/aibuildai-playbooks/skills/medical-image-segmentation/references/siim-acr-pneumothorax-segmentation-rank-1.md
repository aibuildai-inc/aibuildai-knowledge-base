# 1st place solution (with code and configs)

Competition: siim-acr-pneumothorax-segmentation
Rank: #1
Source: https://www.kaggle.com/c/siim-acr-pneumothorax-segmentation/discussion/107824

First of all thanks Society for Imaging Informatics in Medicine (SIIM) and Kaggle for a very interesting and challenging competition. It was a pleasure to participate

This was my first experience with image segmentation, and if someone had told me beforehand how it would all end, I would consider him crazy)

My pipeline: https://github.com/sneddy/kaggle-pneumathorax

## Model Zoo
- AlbuNet (resnet34) from [\[ternausnets\]](https://github.com/ternaus/TernausNet)
- Resnet50 from [\[selim_sef SpaceNet 4\]](https://github.com/SpaceNetChallenge/SpaceNet_Off_Nadir_Solutions/tree/master/selim_sef/zoo)
- SCSEUnet (seresnext50) from \[[selim_sef SpaceNet 4\]](https://github.com/SpaceNetChallenge/SpaceNet_Off_Nadir_Solutions/tree/master/selim_sef/zoo)

## Main Features
### Triplet scheme of inference and validation
Let our segmentation model output some mask with probabilities of pneumothorax pixels. I'm going to name this mask as a basic sigmoid mask. I used triplet of different thresholds: *(top\_score\_threshold, min\_contour\_area, bottom\_score\_threshold)*

The decision rule is based on a doublet *(top\_score\_threshold, min\_contour\_area)*. I used it instead of using the classification of pneumothorax/non-pneumothorax.
- *top\_score\_threshold* is simple binarization threshold and transform basic sigmoid mask into a discrete mask of zeros and ones.
- *min\_contour\_area* is the maximum allowed number of pixels with a value greater than top\_score\_threshold

Those images that didn't pass this doublet of thresholds were counted non-pneumothorax images.

For the remaining pneumothorax images, we binarize basic sigmoid mask using *bottom\_score\_threshold* (another binariztion threshold, less then *top\_score\_threshold*).  You may notice that most participants used the same scheme under the assumption that *bottom\_score\_threshold = top\_score\_threshold*.

The simplified version of this scheme:
```python
classification_mask = predicted &gt; top_score_threshold
mask = predicted.copy()
mask[classification_mask.sum(axis=(1,2,3)) &lt; min_contour_area, :,:,:] = np.zeros_like(predicted[0])
mask = mask &gt; bot_score_threshold
return mask
```

### Search best triplet thresholds during validation 
- Best triplet on validation: (0.75, 2000, 0.3).
- Best triplet on Public Leaderboard: (0.7, 600, 0.3)

For my final submissions  I chose something between these triplets.

### Combo loss
Used \[[combo loss\]](https://github.com/SpaceNetChallenge/SpaceNet_Off_Nadir_Solutions/blob/master/selim_sef/training/losses.py) combinations of BCE, dice and focal. In the best experiments the weights of (BCE, dice, focal), that I used were:
- (3,1,4) for albunet\_valid and seunet;
- (1,1,1) for albunet\_public;
- (2,1,2) for resnet50.
 
### Sliding sample rate
Let's name portion of pneumothorax images as the sample rate.

The main idea is controlling this portion using sampler of torch dataset. 

On each epoch, my sampler gets all images from a dataset with pneumothorax and sample some from non-pneumothorax according to this sample rate. During train process, we reduce this parameter from 0.8 on start to 0.4 in the end.

Large sample rate at the beginning provides a quick start of the learning process, whereas a small sample rate at the end provides better convergence of neural network weights to the initial distribution of pneumothorax/non-pneumothorax images.

### Learning Process recipes
I can't provide a fully reproducible solution because  during learning process I was uptrain my models **A LOT**. But looking back for the formalization of my experiments I can highlight 4 different parts:
- **part 0** - train for 10-12 epoches from pretrained model with large learning rate (about 1e-3 or 1e-4), large sample rate (0.8) and ReduceLROnPlateau scheduler. The model can be pretrained on imagenet or on our dataset with lower resolution (512x512).  The goal of this part: quickly get a good enough model with validation score about 0.835. 
- **part 1** - uptrain the best model from the previous step with normal learning rate (~1e-5), large sample rate (0.6) and CosineAnnealingLR or CosineAnnealingWarmRestarts scheduler. Repeat until best convergence.
- **part 2** - uptrain the best model from the previous step with normal learning rate (~1e-5), small sample rate (0.4) and CosineAnnealingLR or CosineAnnealingWarmRestarts scheduler. Repeat until best convergence.
- **second stage** - simple uptrain with relatively small learning rate(1e-5 or 1e-6), small sample rate (0.5) and CosineAnnealingLR or CosineAnnealingWarmRestarts scheduler.

All these parts are presented in the corresponding experiment folder

### Augmentations
Used following transforms from \[[albumentations\]](https://github.com/albu/albumentations)
```python
albu.Compose([
    albu.HorizontalFlip(),
    albu.OneOf([
        albu.RandomContrast(),
        albu.RandomGamma(),
        albu.RandomBrightness(),
        ], p=0.3),
    albu.OneOf([
        albu.ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03),
        albu.GridDistortion(),
        albu.OpticalDistortion(distort_limit=2, shift_limit=0.5),
        ], p=0.3),
    albu.ShiftScaleRotate(),
    albu.Resize(img_size,img_size,always_apply=True),
])
```
### Uptrain from lower resolution
All experiments (except resnet50) uptrained on size 1024x1024 after 512x512 with frozen encoder on early epoches.  

### Second stage uptrain
All chosen experiments were  uptrained on second stage data

### Small batch size without accumulation
A batch size of 2-4 pictures is enough and all my experiments were run on one (sometimes two) 1080-Ti.

### Checkpoints averaging
Top3 checkpoints averaging from each fold from each pipeline on inference

### Horizontal flip TTA

## Best experiments:
- albunet\_public - best model for Public Leaderboard
- albunet\_valid - best resnet34 model on validation
- seunet - best seresnext50 model on validation
- resnet50 - best resnet50 model on validation

[Experiments dashboard]

## Final Submission
My best model for Public Leaderboard was albunet\_public (PL: 0.8871), and a score of all ensembling models was worse.
But I suspected overfitting for this model, therefore, both final submissions were ensembles.

- The first ensemble believed in Public Leaderboard scores more and used more "weak" triplet thresholds.
- The second ensemble believed in the validation scores more but used more "strict" triplet thresholds.

### Private Leaderboard:
- 0.8679
- 0.8641

I suspect that the best solution would be ensemble believed in the validation scores more, but used more "weak" triplet thresholds.
