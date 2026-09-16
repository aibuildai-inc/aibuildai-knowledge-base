# 10th Place Solution

Competition: ariel-data-challenge-2024
Rank: #10
Source: https://www.kaggle.com/c/ariel-data-challenge-2024/discussion/544189

Thanks to the participants and organizers for this competition. It was very interesting and educational. I hope to see Ariel on kaggle in a year.

## Approach
We used a polynomial approximation approach. Thanks a lot to Sergey for sharing. The main difference of our approach was the use of the entire signal, without cutting out the moments of entry/exit into/from the transit. You can check all the code at this [notebook link](https://www.kaggle.com/code/egorgij21/ariel-final-top10).

## Transit zone detection
The transit phase detection algorithm analyzes the time series of the signal, calculating derivatives in a sliding window to identify sudden changes in light intensity characteristic of the planet entering and exiting the transit state.

Based on the extrema of the derivative, the midpoints of the first and second phases of transit are determined. To clarify the beginning of the first phase, the signal is divided into two linear sections with error minimization, which makes it possible to accurately determine the beginning of the transit. In total, the algorithm returns the entry/exit indices and the duration of the transition to/from transit.


## Function for transit recovery
Having no idea about the general format of a function, multiplication by which could model transit, we chose [function](https://www.desmos.com/calculator/iipeyednvz), which being smooth could approximate the piecewise real one quite well. Further using the UNIFORM format did not lead to an increase in the score, we decided to keep the smooth version.

Transit correction example:


## Prediction Collection Pipeline
Our main pipeline calculates the target by performing wavelength binning aggregation and selecting random sample indices.

Steps:
1) **Calculate Parameters**: We calculate the parameters `p1`, `p2`, and `t`, representing the start of transit, end of transit, and duration of transition to transit, respectively.
2) **Compute d_st and Polynomial on Normalized Signal**: We calculate `d_st`, which represents the mean prediction of the target (denoted as "d" in our code). At this step, we also optimize other parameters obtained in the first step.
3) **Iterate Over Wavelength Indices**: Using a moving window of size `2 * k_binn_wl` and a step of `k_binn_wl // 2`, we iterate over wavelength indices.
4) **Optimize Polynomials on Subsamples**: Within each window, we optimize 15 polynomials on subsamples of randomly selected wavelengths from the current window. For each subsample, we select `k_binn_wl` wavelengths.
5) **Break Down Subsamples**: Each subsample is further divided into smaller wavelength index groups, and we optimize only the target on their averaged signals.
6) **Calculate Predictions for Noised Wavelengths**: Finally, we calculate predictions for noisy wavelengths using the same algorithm, but without the first `k_binn_wl` iterations.
```
p1, p2, t = phase_detector(normalized_planet[:, :-1].mean(axis=1))

t_st, d_st, p1_st, p2_st, poly, deg = calibrate_train_poly(normalized_planet[:, :-1].mean(axis=1), p1, p2, t, x)

for k in range(100, 283, k_binn_wl // 2):
    for j in range(15):
        subsample = np.sort(np.random.choice(np.arange(max(k - k_binn_wl, 0), min(k + k_binn_wl, 283)), k_binn_wl, replace=False))

        signal = normalized_planet[:, subsample].mean(axis=1)

        t_st_j, d_st_j, p1_j, p2_j, poly_j, _ = calibrate_train_poly(signal, p1_st, p2_st, t_st, x, d_st=d_st, best_deg=deg)

        binn_j = 5 * (k // 100 + 1)
        for w_idxs in subsample.reshape((subsample.shape[0] // binn_j, binn_j)):
            signal = normalized_planet[:, w_idxs].mean(axis=1)
            d = calibrate_train(signal, p1_j, p2_j, t_st_j, poly_j, d_st_j, x, method="Nelder-Mead")
            for w_idx_i in w_idxs:
                planet_d[w_idx_i].append(d)

for j in range(100):
    subsample = np.sort(np.random.choice(np.arange(283), 100, replace=False))
    signal = normalized_planet[:, subsample].mean(axis=1)
    t_st_j, d_st_j, p1_j, p2_j, poly_j, _ = calibrate_train_poly(signal, p1_st, p2_st, t_st, x, d_st=d_st, best_deg=deg)

    for w_idx in subsample.reshape((10, subsample.shape[0] // 10)):
        signal = normalized_planet[:, w_idx].mean(axis=1)
        d = calibrate_train(signal, p1_j, p2_j, t_st_j, poly_j, d_st_j, x, method="Nelder-Mead")
        for w_idx_i in w_idx:
            if w_idx_i <= 100:
                planet_d[w_idx_i].append(d)
```
For better understanding, you can check the scheme:

Also for best score we used blending with various k_binn_wl parameter. As it decreases, the noise in the prediction increases, but sometimes the accuracy increases, since it is responsible for the length of the window from which the polynomial of a randomly taken sample will be calculated.

## Sigma estimation
To estimate sigma, we used an ensemble of gradient boostings, which was trained on prediction features. We predicted 2 values ​​for each planet one for "good" wavelengths and one for "noisy" ones. Training was carried out on features from the predictions of one star, and validation on features from the another. BTW good sigma prediction boosted our public score on 0.04 points.

## Results
mean RMSE on train = 4.58e-5
mean RMSE on train for case "the average against all wavelengths" = 5.75e-5

## What didn't work for us
1) **Post pred normaliztion**: We had planned to normalise the polynomials describing the flux after obtaining the pre-transit coefficients so that the subsampling would make more physical sense, but the score for unknown reasons did not increase.
2) **2D Polynoms**: We tried to construct two-dimensional polynomials describing the whole normalised flux, but we could not devote enough time to this approach.
