# Really Simple Approach to Get 39th Place

Competition: ariel-data-challenge-2024
Rank: #39
Source: https://www.kaggle.com/c/ariel-data-challenge-2024/discussion/543682

I would like to thank the host for organizing such a wonderful competition, also to the person who shared a fantastic idea in the discussion, and my teammate, kyu999 and yuto.
We had only one month to work on this competition, but we enjoyed this competition very much.

Our solution is relatively simple.
Our solution consists of wl prediction part and sigma prediction part.

# WL Prediction Part
WL Prediction Part is based on Sergei's method.
We achieved LB 0.572 by this method.
We expanded the Sergei's method to a multiple wavelength prediction version.
Simply predicting by the light curve of each wavelength in AIRS is not enough to expand this method due to the noise.
We reduced noise by taking the sliding window mean along the wavelength axis in AIRS, which made possible to expand this method.
FGS1 data seems noisy, so we predicted wl_1 by the mean of light curves of FGS1 and AIRS.


# Sigma Prediction Part
We utilized Random Forest to predict sigma.
This method boosted LB 0.572 to 0.607.
We wanted to find better approach, but we found this method 2 days before the deadline and we didn't have much time to improve it.:(
We thought that models can overfit easily, so we chose Random Forest.
We used the following features to predict sigma of i th wavelength.
Note that q means MAE/mean(signal during transit) whose shape is (num_planet, num_wavelength), s means wl prediction by the Sergei's method whose shape is (num_planet, num_wavelength), and data means light curve signals whose shape is (num_planet, num_time, num_wavelength).
- q[:, i:i+1]
- s[:, i:i+1]
- std(q, axis=1)
- mean(q, axis=1)
- entropy(abs(q) + 1e-8, axis=1)
- std(s, axis=1)
- mean(s, axis=1)
- skew(data[:, :, i], axis=1)
- kurtosis(data[:, :, i], axis=1)

We simply used 5 fold Stratified KFold for cross validation.


# Deep Learning Approach(Not Worked)
We could not construct good Deep Learning Model better than the LB score 0.531 using CNN.
Validation score fluctuated very much but making batch size large and adding 1e-8 to the sigma prediction made the training more stable.
I'm looking forward to the deep learning solution.

↓our submission code
https://www.kaggle.com/yamashitamotokazu/39th-place-solution-submission-code
