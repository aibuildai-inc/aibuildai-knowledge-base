# 10th place solution

Competition: google-smartphone-decimeter-challenge
Rank: #10
Source: https://www.kaggle.com/c/google-smartphone-decimeter-challenge/discussion/262364

Thanks to kaggle and organizers hosting a nice competition.

## Overview
* Predicting the Noise, `Noise = Ground Truth - Baseline`, like denoising in computer vision
* Using the speed `latDeg(t + dt) - latDeg(t)/dt`  as input  instead of the absolute position to prevent model from overfitting on the train dataset.
* Making 2D image input with Short Time Fourie Transform, STFT, and then using ImageNet convolutional neural network

[pipeline]
[best_vs_host_baseline]

## STFT and Conv Network Part
* Input: Using [librosa](https://librosa.org/doc/latest/index.html),  generating STFT for both latDeg&lngDeg speeds.
    + Each phone sequence are split into 256 seconds sequence then STFT with `n_tft=256`, `hop_length=1` and `win_length=16` , result in (256, 127, 2) feature for each degree. The following 2D images are generated  from 1D sequence.

[stft_images]

* Model: Regression and Segmentation
    * Regression: EfficientNet B3, predict latDeg&lngDeg noise, 
    * Segmentation: Unet ++ with EfficientNet encoder([segmentation pyroch](https://github.com/qubvel/segmentation_models.pytorch)) , predict stft  noise
        * segmentation prediction + input STFT ->  inverse STFT -> prediction of latDeg&lngDeg speeds

        * this speed prediction was used for:
            1. Low speed mask;  The points of low speed area are replaced with its median.
            2. Speed disagreement mask: If the speed from position prediction and this speed prediction differ a lot, remove such points and interpolate.
        * prediction examples for the segmentation.  
       [conv_segmentation_result]
[conv_segmentation_result_2]

## LightGBM Part
  * Input: IMU data excluding magnetic filed feature
      * also excluding y acceleration and z gyro because of phone mounting condition
      * adding moving average as additional features, `window_size=5, 15, 45`
  * Predict latDeg&lngDeg noise

## Public Post Process Part
There are nice and effective PPs in public notebook. I used the following notebooks. Thanks to the all authors.
* [phone mean](https://www.kaggle.com/t88take/gsdc-phones-mean-prediction)
* [filtering outlier](https://www.kaggle.com/dehokanta/baseline-post-processing-by-outlier-correction)
* [kalman filter](https://www.kaggle.com/emaerthin/demonstration-of-the-kalman-filter)
* [gauss smoothing&phone mean](https://www.kaggle.com/bpetrb/adaptive-gauss-phone-mean)

## KNN at downtown Part
similar to [Snap to Grid](https://www.kaggle.com/robikscube/indoor-navigation-snap-to-grid-post-processing), but using both global and local feature. Local re-ranking comes from the  host baseline of [GLR2021](https://www.kaggle.com/c/landmark-retrieval-2020)
* Use train ground truth as database
* Global search: query(latDeg&lngDeg) -> find 10 candidates
* Local re-ranking: query(latDeg&lngDeg speeds and its moving averages) -> find 3 candidates -> taking mean over candidates


## scores
* Check each idea with late submissions.
* actually conv position pred part implemented near deadline, before that I  used only the segmentation model for STFT image.

| status                   | Host baseline + Public PP | conv position pred | gbm | speed mask | knn global | knn local | Private Board Score |
| ---                      | ---                       | ---                | --- | ---        | ---        | ---       | ---                 |
| my best submission       | ✓                         | ✓                  | ✓   | ✓          | ✓          | ✓         | 2.61693             |
| late sub                 |                           |                    |     |            |            |           | 5.423               |
| late sub                 | ✓                         |                    |     |            |            |           | 3.61910             |
| late sub                 | ✓                         | ✓                  |     |            |            |           | 3.28516             |
| late sub                 | ✓                         | ✓                  | ✓   |            |            |           | 3.19016             |
| late sub                 | ✓                         | ✓                  | ✓   | ✓          |            |           | 2.81074             |
| late sub                 | ✓                         | ✓                  | ✓   | ✓          | ✓          |           | 2.66377             |


## code
* My code is available [here](https://github.com/Fkaneko/kaggle_Google_Smartphone_Decimeter_Challenge)
* My all pretrained weights have been uploaded at [kaggle dataset](https://www.kaggle.com/sai11fkaneko/google-smartphone-decimeter-challenge-weights)
* You can reproduce my result with the [instruction](https://github.com/Fkaneko/kaggle_Google_Smartphone_Decimeter_Challenge#how-to-run)
