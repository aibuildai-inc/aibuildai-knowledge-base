# 19-th place write up

Competition: LANL-Earthquake-Prediction
Rank: #19
Source: https://www.kaggle.com/c/LANL-Earthquake-Prediction/discussion/94450#latest-544035

Thank you for organizers and congratulations to all the participants, it was very difficult competition with unstable public LB. Also thank you to team mates for tackling this difficult problem together.

We have chosen 2 final submissions as 

(A). Scale ttf value based on the test dataset ttf distribution prediction given in the [discussion](https://www.kaggle.com/c/LANL-Earthquake-Prediction/discussion/90664#latest-535844)
  - original prediction is done by GBDT models (XGBoost, LightGBM, CatBoost)

(B). Ensemble of 5 models, without ttf rescaling.
  - GBDT models (XGBoost, LightGBM, CatBoost) + NN models (1D CNN + 2D CNN)

Scores were (A): 2.401 and (B): 2.599 respectively.

I was mainly working on NN part in the team, so I will write up mainly 2nd submission. (As a result, 2nd submission score was quite low compared to 1st submission, so this is just as a write up, NOT the solution (A) that wins 19-th place).
Solution (A) will be posted by other team mates later!
[EDIT] Please refer [19th place solution (GBDT + post-processing)](https://www.kaggle.com/c/LANL-Earthquake-Prediction/discussion/94594)

## CV strategy
We were validating using Group K-fold where group id is assigned to each quake.
It returns quite unstable result in each fold, because the validation dataset quake ttf distribution differs a lot.
Later in the competition we splitted quake id manually to distribute ttf as even as possible to make training a little bit more stable.

## NN model
We used 2 models.
 - 1D-CNN: directly handle 150000 data is difficult in terms of computation power. So the feature is calculated by sliding window to make small 1d series data to process on 1d cnn.
 - 2D-CNN: 150000 point data is converted to 2d array using STFT, and process it on 2d cnn.


## Data processing
 - high pass filter --&gt; wavelet denoising
 - clip acoustic data value to (-20, 20): surprisingly, it seems that even we clip the big value, the loss still decreases and clipping value makes the feature looks more stable.

### artifact classification
We created a classifier which predicts 4096(4095) record jump by simple 2-layer MLP model given window width=20.
We could get high accuracy to predict record point, this is used to determine start point of STFT preprocessing for 2D-CNN.

### Remove after quake segment
Even after quake happens (big acoustic_data shake ends) some segments (around 3 segment) have ttf = 0.
It may be outlier to make training difficult, and I removed these data from training. 

## Prediction target
For 2D-CNN, we trained following 3 labels at the same time.
 - ttf: time to failure
 - tsf: time since failure
 - tqt: total time quake (ttf value at the beginning of quake)

tsf and tqt feature is calculated from ttf information.
I thought learning multiple label works as regularizing effect. But it was not so significant to the performance.

## Data augmentation
I thought it is quite important to reduce overfitting in this competition, so tried many kinds of data augmentation methods.

### We tried and seems worked a bit (adopted)
 - adding noise
 - flip (flip along mean)
 - time flip
 - cutout on 1d raw wave data
 - cutout on STFT time domain

### We tried but not worked (not adopted)
 - cutout on STFT frequency domain
 - wave shift (enlarge/shrink on time domain)

## Model pretraining with p4581
We wanted to utilize p4581 data. After I checked the data, it contains too long quake or too short quake. So I removed quake data with quake time &lt;6sec and &gt;18sec.
2D-CNN model is trained with this data, and its weight is used to fine-tune with competition dataset.
Actually its effect to performance was not so much.

## Hyper parameter tuning
We used [optuna](https://github.com/pfnet/optuna) for hyper parameter tuning.

## Training performance
We trained model with 5-group K fold CV with different 5 seeds (total 25 models).

 - 1D CNN model: CV oof MAE around 1.93
 - 2D CNN model: CV oof MAE around 2.00

## Ensemble/Stacking
We tried 2 ways for ensembling model's prediction
 - simply taking mean (submitted (B)): private LB 2.522
 - stacking using `BayesianRegression` (not submitted): private LB 2.599
