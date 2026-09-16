# 16th place solution

Competition: rsna-2023-abdominal-trauma-detection
Rank: #16
Source: https://www.kaggle.com/c/rsna-2023-abdominal-trauma-detection/discussion/447448

Thanks to the organizers for such a great competition. Unfortunately, I couldn’t take part last year in a similar one due to the lack of hardware. However, this year is different, and I’m really happy with the results that we managed to achieve. 

## Problem
In this competition we were tasked with predicting the intensity of injuries for different abdominal organs. The available data consists of big CT images (in DICOM format) with partial supplemental segmentation annotations (in NIFTI format).

In ML terms, all comes down to 3D segmentation / classification models, lack of data / annotations, and heavily-penalizing metric.

## Summary
- **U-Net Bi-Conv-LSTM** segmentation for organs (kidney, liver, spleen, bowel) and separate model for Extravasation based on boxes from here (thanks a lot!)
- **Resnet 3D CSN** for 3D crops separated into 2 stages: (kidney, liver, spleen) and (bowel)


## 3D Semantic Segmentation

The semantic segmentation part to identify organs is quite straightforward (compared to the later classification) and could be effectively performed by **U-Net Bi-Conv-LSTM** with Effnet_v2_b0 backbone and **CE-Dice-Focal** loss. 

To make things efficient, we train semantic segmentation on **96x256x256 crops** and predict the whole image using crops of size 96x256x256 with **overlaps of 48x256x256** (later on overlaps were removed to save time and space in inference).

To elevate overfitting (it’s not that critical, especially compared to classification), we added **geometric augmentations** like ShiftScaleRotate, RandomBrightnessContrast, Flips, GridDistortion, ElasticTransform. 

On average, about 10% of total volume was dedicated to kidney, liver, spleen, and about 20% - to bowel.

The **macro dice score per image** is around **0.96**. 

**Training time** - about **12 hours**.

## 3D classification

Based on extracted crops from segmentation masks, we train 2 models: one for kidney, liver, spleen, one for bowel. 

The **CSN models** from mmaction proved to be very fast and accurate. In order to figure out how to deal with temporal dimension, several possibilities were explored, but in the end basic interpolation (**3D resize**) was used to convert crops to **96x256x256 resolution**.

To battle overfitting (which is really severe even with CSN), **intensive geometric augmentations** were used, including ShiftScaleRotate, RandomBrightnessContrast, and 4 different types of Blurs.

The mean competition loss across all folds is **0.401** for kidney, liver, spleen and 0**.156** for bowel.

**Training time** - about **4 hours** per kidney, liver, spleen fold; and **8-10 hours** per bowel fold. 

## 3D classification for Extravasation

In order to make predictions for Extravasation, a segmentation model was utilized. The motivation is simple: if semantic segmentation model predict anything, there is Extravasation, and it should be reflected in the probabilities. 

To make the **transition from semantic segmentation to classification**, the following trick was used:
- Turn 3D mask to 1D 
- Sort probabilities 
- Take top_n probabilities
- Find mean values of them. That’s the probability for positive Extravasation.

In pseudo-code:
`cls_pred = np.mean(np.sort(np.ravel(sigmoid(mask)))[::-1][:top_n])`

The mean log loss across all folds is **0.543** for extravasation, and **0.501** for any_injury. 

**Training time** - about **4 hours** per fold. 

## Validation

**StratifiedGroupKFold** (stratification based on classification labels, grouping based on patients) with 4 folds. 

Mean log loss across all folds and all groups (kidney, liver, spleen, bowel, extravasation, any_injury), (which is the **competition metric**) is **0.400**. 

## Additional tricks

- No post-processing.
- SWA on final checkpoints.
- EMA during training.
- Temporal shifting in classification to battle overfitting even more.
- Gradient checkpointing to have bigger batches (important for classification).
- memmap (uint8) using numpy to speed-up data reading and crop extraction. 
- 2 final subs: one minimizing competition loss, one maximizing AUC


## Things that didn’t work
- Samplers
- Heavier models (2+1D or Uniformer)

## Final notes 
During the final 2 days of the competition, we managed to improve the models for kidney, liver, and spleen from **0.4** to roughly **0.38**, which brought the overall loss from **0.4** to **0.39**, but made some errors in the submission process, which made them useless. The trick is simple - increase batch size. Usually we train with the batch of 14, but could increase it to 24 (with the help of A100 cards). 

The total **training time** (including all 4 folds for each stage) is around **80 hours** using a single RTX A6000 Ada.

The total **submission time** is around **8-9 hours** using a single Tesla P100. 

We believe this solution could be pushed much further. However, we made the first sub (that isn’t sample submission or just a bunch of static predictions) 2 days before the competition ended, so that also played some role.

P.S. Man that sucked to mess up the models for 0.39 :)
