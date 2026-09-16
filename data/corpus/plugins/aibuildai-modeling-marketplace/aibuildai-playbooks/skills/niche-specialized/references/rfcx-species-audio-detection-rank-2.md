# 2nd place solution

Competition: rfcx-species-audio-detection
Rank: #2
Source: https://www.kaggle.com/c/rfcx-species-audio-detection/discussion/220760

## Overview

I trained simple classification models (24 binary classes) with logmel spectrograms :
1. bootstrap stage: models are trained on TP/FP with masked BCE loss
2. generate soft pseudo labels with 0.5 second sliding window 
3. train models with pseudo labels and also sample (with p=0.5) places with TP/FP - this partially solves confirmation bias problem. 

Rounds of pseudo labeling and retraining (points 2,3) were repeated until the score on public LB didn't improve. Depending on the settings it took  around 4-10 rounds to converge.

My initial models that gave 0.86 on TP/FP alone easily reached 0.96x with pseudo labeling . After this success I gave this challenge a 5 weeks break as I lost any motivation to improve my score :)
Later to my surprise it was extremely hard to beat 0.97 even with improved first stage models.


## Melspectrogram parameters
- 256 mel bins
- 512 hop length
- original SR
- 4096 nfft

## FreqConv (CoordConv for frequency)
After my first successful experiment with pseudo labeling that reached 0.969 on public LB  I tried to just swap encoders and blend models but this did not bring any improvements. 
So I visualised the data for different classes and understood that when working with mel spectrograms for this task we don’t need translation invariance and classes really depend on both frequency and patterns.
I added a channel to CNN input which contains the number of mel bin scaled to 0-1. This significantly improved validation metrics and after this change log loss on crops around TP after the first round of training with pseudo labels was around 0.04 (same for crops around FP). Though it only slightly improved results on the LB.


## First stage 

For the first stage I used all tp/fp information without any sampling and made crops around the center of the signal.

**Augmentations**
- time warping
- random frequency masking below TP/FP signal
- random frequency masking above TP/FP signal
- gaussian noise
- volume gain
- mixup on spectrograms

For mixup on spectrograms - I used constant alpha (0.5) and hard labels with clipping (0,1). Masks were also added. 

## Pseudolabeling stages
Sampled TP/FP with p=0.5 otherwise made a random crop from the full spectrogram. 

Without TP/FP sampling labels can become very soft and the score decreases after 2 or 3 rounds.
After training 4 folds of effnet/rexnet I generated OOF labels and ensembled their predictions. Then the training is repeated from scratch.

**Augmentations**
- gaussian noise
- volume gain
- mixup 
- time warping
- spec augment
- mixup on spectrograms

**Mixup**

I used constant alpha (0.5) and added soft labels from two samples. This hurts logloss on FP a bit but at the same time significantly increases recall on TP.

## Validation
	
Local validation did not have high correlation with the public leaderboard. Logloss on TP was somehow correlated but still it was not robust. 
So without proper validation I decided to not select the best checkpoints and just trained 60 epochs (around 200 batches in each epoch) with CosineLR and AdamW optimizer.

My best models on validation - auc 0.999, log loss 0.03 did not produce great results (0.95). After the competition though It turned out that they can be easily improved with  postprocessing to 97x-98x range.


## Final ensemble

I used 4 models with 4 folds from Effnet and Rexnet (https://arxiv.org/abs/2007.00992  lightweight models with great performance) families:
- Rexnet-200 (4 sec training/inference), EffnetB3 (4 sec training/inference)
- Rexnet-150 (8 sec training/inference), EffnetB1 (8 sec training/inference)

Rexnet was much better than EfficientNet alone (less overfitting), but in ensemble they worked great.

During inference I just used 0.5 second sliding window and took max probabilities for the full clip and then averaged predictions from different models.

## Lessons learned
I did not know about the paper and lacked this useful information about the dataset.

In my solutions I often rely on models alone but don’t explore the data deeply. 
In this case I understood that the relabeled train set has similar class distribution to the test set and decided that models would easily learn that. I was wrong and simple post-processing could significantly improve results (though this happened due to severe class imbalance).
