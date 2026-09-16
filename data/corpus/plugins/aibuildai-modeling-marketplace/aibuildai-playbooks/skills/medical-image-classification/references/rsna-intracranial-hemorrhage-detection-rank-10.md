# 10th place solution (+ github code)

Competition: rsna-intracranial-hemorrhage-detection
Rank: #10
Source: https://www.kaggle.com/c/rsna-intracranial-hemorrhage-detection/discussion/118873

Congratulations to all participants and the winners ! 
And, I became KaggleMaster on this competition ! 
We will go to Japanese BBQ (Yakiniku) by prize money of this competition, haha.

Following is a summary of our solutions.

Code : https://github.com/shimacos37/kaggle_rsna_2019_10th_solution
We mostly used @appian code. Thank you very much @appian !!

#  Pipeline

.png?generation=1574649511734185&amp;alt=media)

#  Summary

## Stage 1
### Preprocess

- As most of people did, we applied three window (brain, blood/subdural, bone).
- Delete some noisy image (image which has small brain area).
- PatientID based 5-fold split.

### Train

- We trained simply changed backbone in @appian code and applied some ideas.
- We usually used 512x512 img_size and applied simple augmentations (flip, resize, etc...)
- Finally, We constructed eleven models. Consequently, I think that it doesn't need to construct so many models...

#### Simple CNN models

1. SeResNext-50
1. SeResNext-50 (Resize 410x410)
1. SeResNext-101 (Mixup used)
1. Efficientnetb3
1. InceptionV4
1. InceptionResNetV2
1. Xception

#### Some Ideas

- We predicted label without 'any' and 'any' by other label probability (1 - (1-p_1)*(1-p_2)...)
  - This is not so high score, but should be have some contribution when stacking.

- We used adjacent images for input, and predict center label. Please see following figure.

.png?generation=1574650665165165&amp;alt=media)

- We applied label smoothing by moving average or interpolation of the sandwiched label area.
  - Because, we noticed the the boundary of label tends to have high log_loss by our EDA.

## Stage 2 

### Preprocess

- First, we predicted the probabilities of labels per an image.
- Second, we sorted the probabilities by Position2 per StudyInstanceUID.
- We extracted below features.
  - Aggregate feature (min, max, mean, std), pred-pred_mean, pred / pred_mean, etc
  - Moving average feature (3, 5, 7, 9 adjacent prediction), pred - moving_average_pred, pred / moving_average_pred, etc

### Stacking

- We simply trained LightGBM and MLP by above features.
- And we constructed CNN stacking model like below figure.

.png?generation=1574658076805462&amp;alt=media)

- We treated above features as images.
  - height : features from different models
  - width : feature dimension
  - channel : adjacent features sorted by Position2

## Stage 3

### Preprocess

- We used the same method of Stage 2.

### Stacking

- We simply trained LightGBM
- We clipped prediction values by [1e-6  1 - 1e-6] and made submissions.
