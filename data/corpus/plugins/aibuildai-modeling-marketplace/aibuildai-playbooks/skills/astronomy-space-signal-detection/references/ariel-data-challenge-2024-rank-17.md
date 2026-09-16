# 17th Parametric Fitting Approach

Competition: ariel-data-challenge-2024
Rank: #17
Source: https://www.kaggle.com/c/ariel-data-challenge-2024/discussion/543763

When I entered this competition, I began with a classical fitting approach for signal reconstruction. Surprisingly, it performed very well compared to other models I tested. This approach has several advantages: it is stable in noisy data, relatively simple to implement, and offers a way to construct reliable confidence intervals.

The method consists of two main steps: data detrending and transit model fitting.

**Data detrending:**
In this step, we identify transit breakpoints, mask the transit phase, and fit a quadratic polynomial to the remaining data. This parametrization provides a model that describes the star’s flux in absence of a planetary transit. The detrended data from this model is used in the next step (see figure).

**Transit parametrization using the `Erf` function:**
I found the error function is great for catching the transition zones and offers an estimate of transit depth via its vertical offset. The fitted model consists of two error functions with opposite signs, representing the ingress and egress of the transit. It is quite stable even if the detrending of data is not perfect. The fit errors are scaled to ensure that the confidence intervals cover approximately 99% of cases



The next step is about testing sensitivity to the spectrum.

**Choice of wavelength steps:**
Predicting the transit depth at every individual wavelength point can be noisy, so two scenarios were considered for predictions:
1) Merging data by averaging every 15 wavelength values
2) Using a single, wavelength-inclusive result for cases with low sensitivity.
If both scenarios align within a specific p-value threshold, I use the wavelength-inclusive result (2) for prediction. Otherwise, I rely on the merged data scenario (1).
