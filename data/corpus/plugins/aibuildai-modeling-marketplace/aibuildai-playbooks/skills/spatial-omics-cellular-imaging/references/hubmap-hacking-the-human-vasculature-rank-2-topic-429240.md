# 2nd place solution

Competition: hubmap-hacking-the-human-vasculature
Rank: #2
Source: https://www.kaggle.com/c/hubmap-hacking-the-human-vasculature/discussion/429240

Many thanks to Kaggle and HuBMAP who organized another very interesting competition. This is a joint writeup with my long-time teammate @phamvanlinh143 , who did most of the hard work in this one.

**Summary**

This competition is an especially tricky one with several problems that needed to be addressed:
-	Public and private test set are structured differently, with private set coming from an unseen WSI. Public test set size is also quite small. With these two reasons combined we have a completely unreliable LB.
-	The strange effect of dilation on public LB result.

Fortunately, we came up with a strategy that we believed can deal with each of these issues accordingly:
-	Build a reliable CV and trust it.
-	Train a large and diverse ensemble for stability.
-	Submit the ensemble with dilate and without dilate for our two submissions.

**Cross-validation**

To build a trustworthy CV, it must follow as closely as possible to how private test set is created. That means:
-	Validation must be done on dataset 1 labels.
-	Train and validation set must not contain the same WSI.

A simple solution for this is to use Dataset1-WSI1 as one validation fold and Dataset1-WSI2 as the other, but this would leave out a lot of training samples, especially if we want to train using only Dataset 1. 

What we did is using metadata to split the Dataset1 tiles from each WSI into left and right side, resulting in 4 folds:
-	Dataset1 – WSI1 - Left
-	Dataset1 – WSI1 - Right
-	Dataset1 – WSI2 - Left
-	Dataset1 – WSI2 – Right

Next, for each tile in training data we use `staintools` to generate 9 according tiles transformed into the style of the 9 additional WSIs in Dataset 3. In training, when the sampled tile come from the same WSI of the validation set, one of the 9 generated tiles are sampled instead. While this method doesn’t completely remove the characteristics of the original WSI, we felt that it is a good enough compromise to what we wanted to achieve.

An example of an original tile and its 9 variations:


Splitting the dataset in this manner also allowed us to train models using tile concatenation with minimal leak, as we only had to handle the middle tile column that separated left and right side of the WSI.

**Model training**

**Input data:**  we used one of these two data types:
-	original tiles
-	padded tiles similar to what @hengck [proposed](https://www.kaggle.com/competitions/hubmap-hacking-the-human-vasculature/discussion/419143#2316842). We padded 128 pixels around the original tile using available neighboring tiles. Instance labels are modified accordingly. At inference time we predicted on padded region and then center crop.

An example of original tile (left) and padded tile (right):


**Augmentation:** we applied strong augmentation:
-	stain augmentation: p=1.0 for tiles from same WSI as validation set, p=0.5 otherwise
-	RandomRotate90, RandomFlip, ElasticTransform, ShiftScaleRotate, RandomBrightnessContrast, HueSaturationValue, ImageCompression, GaussNoise, GaussianBlur…
-	AutoAugment similar to DETR [training config](https://github.com/open-mmlab/mmdetection/blob/master/configs/detr/detr_r50_8x2_150e_coco.py).

**Training details:** we trained the models in 2 stages, using only blood vessel class. 
-	Stage 1: train for 30 epochs using 3 folds dataset 1 + all dataset 2
-	Stage 2: finetune for 15 epochs using only 3 folds dataset 1

Then we picked 5 checkpoints from finetune stage and do SWA to provide the final model.

**Models:** Cascade Mask-RCNN models with `swin-t`, `coat-small`, `convnext-t` and `convnext-s` backbones. We used mmdet 2.x to train our models.

We experimented with different combination of backbones and input data types (original or padded). We selected the models with best CV and trained a version using full data. The final ensemble consists of both fold models and full data models.

**Postprocessing:** The masks that met following criteria are filtered:
-	Glomeruli filter: remove masks with more than 60% area inside glomeruli regions.
-	Confidence filter: remove masks with confidence <0.01.
-	Ensemble filter by pixel: count for each pixels the number of blood vessel prediction by the ensemble (how many models predicted positive for that pixel). A threshold is then calculated from the resulting list of pixel counts with quantile q=0.05 (ignore zero value). Masks of which all pixels had count smaller than this threshold are removed.
-	Ensemble filter by instance: perform nms with iou_thresh=0.65. For each selected mask we save the number of masks that satisfied iou_thresh with it. Similar to pixel counts filter, we remove masks with small number of overlapping masks using quantile q=0.075.
-	Small mask filter: remove masks with fewer than 64 pixels.

**To dilate or not to dilate**

As others have reported, we observed significant change of score on Public LB with and without dilation. As adding dilation didn’t work at all on our cross validation, we knew we cannot trust it. However there exists also the probability that private set would present the same label patterns as public set, as the host have given confirmation that they were verified using the same procedure. Fortunately, we had two submissions, so this is where we decided to put it to good use.

Our final ensemble scored: 
-	With dilation: 0.551 public, 0.526 private.
-	Without dilation: 0.465 public, 0.588 private.

Our late submissions showed that submitting any single backbone in our ensemble without dilation would have landed us in the private gold zone anyway while scoring 0.4x on public LB. Even though we chose correctly by trusting our CV, this public LB behavior is still a mystery to us. Hopefully we can shed some light into it by reading the solutions of other top teams.

Thank you very much for reading and let us know if you have any questions.

Edit: 
- training code: https://github.com/phamvanlinh143/HubMap_2023_2nd_Place_Solution
- inference notebook: https://www.kaggle.com/code/phamvanlinh143/hubmap-2nd-place-inference
