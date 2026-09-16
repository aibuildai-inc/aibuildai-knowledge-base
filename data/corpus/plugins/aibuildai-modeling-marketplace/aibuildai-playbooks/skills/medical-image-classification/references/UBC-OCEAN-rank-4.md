# 4th place solution

Competition: UBC-OCEAN
Rank: #4
Source: https://www.kaggle.com/c/UBC-OCEAN/discussion/465811

## TMA
TMA images are centre cropped (eg. 3000 -> 2500) and resized to 768x768 pixels.

## WSI
A single segmentation model is used for title selection.  Segmentation is trained on thumbnail images and supplemental mask data. Tiles from thumbnails are used for mask generation. The location of the pixel with highest probability of being cancerous is selected on the WSI and a region of 1536x1536  pixels around it is cropped and resized to 768x768 pixels, which is then used to predict scores.

## Models
- A total of 16 models based on Convnext, Hornet, Efficientnetv1, Efficientnetv2 are used. Instead of softmax,
sigmoid activation is used for predicting scores.
- Models are trained on 5 labels + non-cancerous label (using supplemental mask data).
- Loss used: Binary crossentropy.
- Augmentations used: Stain augmentation, scaling, rotation, flipud, fliplr, random contrast, random brightness, and random hue (thought it might work as stain augmentation).
- Median averaging is used for generating score.
- A single classifier model along with the segmentation model gives a score of 0.49, 0.55 for public and private leaderboard respectively. 
- Models are divided among the two gpus (T4 x 2) for memory efficiency.

## External Data
No external data was used.

## Outliers
Prediction with low scores (< 0.05) can be labelled  as *Others*. Another method is to predict bottom 5 or 10 percentile scores as *Others*.

## Code
[Submission notebook](https://www.kaggle.com/code/mmelahi/ubc-ocean-final-inference/notebook)
[Submission notebook - single model](https://www.kaggle.com/mmelahi/ubc-ocean-final-single-model-inference)
[Github](https://github.com/ManzoorElahi/UBC-Ovarian-Cancer-Subtype-Classification-and-Outlier-Detection)

## Problematic thumbnail images


**5251_thumbnail.png**

When two or more slices are added side by side as in the above image, the height of the thumbnail becomes smaller, making it harder for the model to predict accurately.  For such thumbnails, new thumbnails using WSI are generated - this boosted my score significantly.
