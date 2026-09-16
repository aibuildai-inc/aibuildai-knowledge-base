# 7th Place solution

Competition: ariel-data-challenge-2024
Rank: #7
Source: https://www.kaggle.com/c/ariel-data-challenge-2024/discussion/543679

Thanks ARIEL team for hosting this competition it was a lot of fun.
Also shoutout to all the amazing public notebooks. They were very helpful to get me started.

Short Summary:
No Deep Learning
Transit depths were fitted using Sergei's polynomial regression method https://www.kaggle.com/code/sergeifironov/ariel-only-correlation.
**Transit Detection**
1. Compute first derivative over entire signal. Convolution window of 30 over the first derivative. Find the min and maxes, those are the two critical points. Then fit a polynomial and transit depth with a fixed ingress/egress window size.
2. Count points from the critical points until the points get within 1-3 std of the fitted polynomial. This is a more precise estimate of the ingress/egress duration. I then proceed to find the transit using the updated transit regions.

**Predicting Mean Transit**
-average wavelengths 0 to 240
-used gaussian filter on each region separately.
-averaged the prediction of polynomial fitting with deg 2 to deg 4. If there was too much curvature, I use deg 4 instead.

**Basic data processing.** They didn't affect cv at all, but gained me ~20 points on public/private leaderboard
- Filled nan values using the average of symmetry across spatial column and interpolated flux between the two closet wavelengths (in cv interpolation method was way better, but worse in the public leaderboard)



**Predict individual wavelengths**
-Convolved the signal across wavelength axis with window size 20 to 60, based on the MAE of the signal. So, if the MAE is lower, I can afford to have a smaller convolution window.
- The individual wavelength predictions were shifted by the mean. And sometimes scaled down.

**Sigma (estimated by scaling off two components)**
-based on the MAE of the residuals between fitted polynomial curve and unprocessed signal. This MAE is also normalized by the fitted curve so that it is in the same units as Transit Depth (percentage)
-based on the "flatness" of the spectra curve. The reason for this is I convolved the signals across wavelengths. So, the flat spectra curves estimates are going to have higher certainty.

**Predicting flat curves**
- I saw in the training data that most of the spectra curves were flat. So, I sought to predict whether the spectra curve was flat or not.
If it was, I predict the mean transit and set a very low std (also based on the two components I mentioned earlier).

Link to code
https://www.kaggle.com/code/junglebeastds/ariel-data-challenge-2024-submission-v7?scriptVersionId=204717619
