# 5th place solution

Competition: UBC-OCEAN
Rank: #5
Source: https://www.kaggle.com/c/UBC-OCEAN/discussion/466017

First of all, I would like to express my gratitude to the Kaggle staff for hosting this fantastic competition, as well as to the organizers at UBC. I also want to extend my appreciation to all the hardworking participants who dedicated themselves to this competition. Special thanks go to my teammates @kapenon and @tascj0 for their tireless efforts and insightful discussions.

# Solution Overview
## Summary of submitted final solution:
1. Inference contains two stages:
  1. Tile selection model -> Only WSI
  2. Classification model -> WSI and TMA

## Training details of segmentation model (mainly WSI, TMA is simply center-cropped):
1. Tile classification helper model (Not used for inference)
  1. Random cropped tiles at 1536x1536, background excluded
  2. Augmentations:  Random horizontal and vertical flips, RandomRotation, RandAugment, RandomGrayscale, RandomErasing
  3. WSI label as tile label
  4. ConvNeXt-base
  5. 6-class classification (Hubmap external data as “Other”)
2. Segmentation helper model (Not used for inference)
  1. Use host provided mask
  2. Tumor binary classification
  3. 2x magnification
  4. SEResNeXt101 UNet
3. Tile selection segmentation model (Used for inference)
  1. Label generation
     1. Inference all tiles using 1st step models and save predicted probabilities of WSI class
     2. Inference all WSIs using 2nd step models
     3. Create a heatmap at 2x magnification
     4. Heatmap GT is 0.5 classification confidence + 0.5 tumor confidence
  2. Training
     1. SEResNeXt101 UNet


Generated heatmap example:[heatmaplabel]
Inference example of segmentation model:

## Training details of tile classification model (Used for inference)
  1. Random cropped tiles at 1536x1536, background excluded
  2. Augmentations: Random horizontal and vertical flips, RandomRotation, RandAugment, RandomGrayscale, RandomErasing + StainNorm
  3. WSI label as tile label
  4. ConvNeXt-base, ConvNeXt-large, EVA (at 448x448)
  5. 6-class classification (Hubmap, Camelyon16, Camelyon17, etc. tiles as “Other”)
  6. Data mining:
     1. Train 1st round
     2. Predict all foreground tiles
         1. Confidence <0.3 tiles are pseudo labeled “Other” in 2nd round
         2. Confidence 0.3-0.6 tiles are ignored
         3. Confidence >0.6 tiles are pseudo labeled as WSI label
     3. Train 2nd round


## Inference details:
### WSI
1. Tile selection
  1. Predict heatmap at 2x magnification, top-5 confidence tiles are selected for classification
2. Classification
  1. Prediction 5 tiles and average predictions as WSI prediction
  2. StainNorm as TTA
3. There are some handling of large WSI images, refer to the submission notebook
### TMA
1. Center crop 1 3072x3072 tile
2. Resize to 1536x1536
3. Predict using the classification model
4. StainNorm as TTA


## Used External Datasets and their license:
1. Hubmap: HuBMAP + HPA - Hacking the Human Body (https://www.kaggle.com/competitions/hubmap-organ-segmentation/overview)
2. Camelyon16, 17: CC0 (https://www.google.com/url?q=https://camelyon17.grand-challenge.org/Data/&sa=D&source=docs&ust=1704419788737632&usg=AOvVaw1BeJOZcucbo9R0wgu5NK3q)
3. ovarian-bevacizumab-response (only for WSI labeled as 'Other'): CC BY 4.0 (https://www.google.com/url?q=https://www.cancerimagingarchive.net/collection/ovarian-bevacizumab-response/&sa=D&source=docs&ust=1704419788739913&usg=AOvVaw0hQlYStI1_qpi7N_YIuHhY)

## What worked
1. Stain normalization.
2. Data mining (pseudo labeling).
3. Segmentation model for identifying valuable patches.
4. Using external datasets as ‘Other’.
5. ConvNext and EVA02 models.
6. Multiscale ensemble.

## Codes
1. Train: https://github.com/ShuzhiLiu/UBC-OCEAN_5th_solution
2. Submission Notebook: https://www.kaggle.com/code/liushuzhi/5thplacesolutionsubnotebook?scriptVersionId=158180788
  1. Without EVA02: Public=0.61, Private=0.61 -> Selected for final sub
  2. With EVA02: Public=0.60, Private=0.63 -> Not selected due to the lack of local CV


## Acknowledgement
We would like to express our gratitude to the Kaggle support system and the emotional support of Rist inc.
