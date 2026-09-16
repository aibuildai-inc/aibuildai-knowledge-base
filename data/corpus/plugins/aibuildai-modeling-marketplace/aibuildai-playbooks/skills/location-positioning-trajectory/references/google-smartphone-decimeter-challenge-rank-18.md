# 18th Place Solution

Competition: google-smartphone-decimeter-challenge
Rank: #18
Source: https://www.kaggle.com/c/google-smartphone-decimeter-challenge/discussion/261774

Congratulations to all the winners, and thanks so much for hosting such an interesting competition! 
This competition is very tough for me, I'll share our solution.

## Overview
[GSDC-solution]



## 1. Reproduce Baseline
In this competition, a baseline using WLS was shared(result file only). In order to reproduce this, we reproduced the baseline using derived files. It was difficult for us to reproduce the exact same results, but we were able to create baseline equivalent scores and decided to blend them with the original baseline file.

We refered this notebook.
https://www.kaggle.com/hyperc/gsdc-reproducing-baseline-wls-on-one-measurement

Here are some of the things we did to create the baseline
- Use calculated satellite(GPS only) position by using OSR data(but effect is a little)
- Use only GPS/GALILEO/QZS data(other satellite is not good)
- Weight of WLS are hand-tuned (x**2 + 3x)

After creating two new baseline locations using the above method, we blended them with the original baseline by taking a weighted average.

## 2. Area Classification
We automatically classified the collection into three categories as in the public version, and also defined a difficult area.
The difficult area was defined as the area like following.
1. Apply pre processing to train data
2. Extract point which error is more than 5m
3. Convert point to polygon by apply buffer to each point
4. Define these polygon area as difficult area

Defined difficult area are like this.
[スクリーンショット 2021-08-05 9 36 09]

This was used for snap to grid.

## 3. Pre/Post Processing
We used some shared notebook, Thanks you to the contributor.

- Outlier Correction
https://www.kaggle.com/dehokanta/baseline-post-processing-by-outlier-correction

- Kalman Smoothing
https://www.kaggle.com/emaerthin/demonstration-of-the-kalman-filter
We apply linear interpolation to keep epoch width constant

- Phone Mean
https://www.kaggle.com/t88take/gsdc-phones-mean-prediction
https://www.kaggle.com/bpetrb/adaptive-gauss-phone-mean

- Remove Phone
https://www.kaggle.com/columbia2131/device-eda-interpolate-by-removing-device-en-ja

- Snap to Grid
We apply snap to grid only downtown area and difficult area.

- Stop Mean
I tried to take the average of the stopping points.
The procedure is as follows.

(1) Predict car speed by lightGBM
  - target: speedMps in ground_truth.csv
  - features
    - lag features with shift range is -30 ~ 30 (location, time, speed etc...)
    - aggregate features

(2) If the prediction result is less than 0.95m/s and more than 2 consecutive seconds, make a group.
(3) Take the average for each group


## 4. Position estimation by imu data
Relative position correction was performed using IMU data for the only downtown area.
I refered this notebook. Thanks.
https://www.kaggle.com/alvinai9603/predict-next-point-with-the-imu-data

- model: lightGBM
- GroupKFold(group="phone")
- features
  - lag features(shift range is -30~30)
  - aggregate features(mean, std, max, min, median, skew, kart)
