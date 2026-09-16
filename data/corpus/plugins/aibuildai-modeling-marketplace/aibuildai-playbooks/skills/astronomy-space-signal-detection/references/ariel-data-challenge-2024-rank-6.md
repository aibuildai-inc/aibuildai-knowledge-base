# 6th place solution

Competition: ariel-data-challenge-2024
Rank: #6
Source: https://www.kaggle.com/c/ariel-data-challenge-2024/discussion/543666

It was an excellent competition! For many years, I've been following astrophysics news with great interest. What else will they see in these blurry pixels? When will they finally find a habitable planet? And now I've been lucky enough to contribute to this myself.

I want to thank the competition hosts, the Ariel mission, and specifically @GordonYip for answering important questions. I'd also like to thank all the authors of the notebook with the correct data calibration process. Without this, my participation would hardly have been meaningful. It's so difficult to find information about calibrations in articles! Thank @shlomoron for the meme thread and funny comments 😀

My solution isn't particularly sophisticated. Here's the breakdown:

**Data Processing**:
I noticed outliers in pixel data and smoothed them with a Gaussian in the frequency axis direction. The "extra" wavelengths not included in the final spectrum were somewhat useful here. Consistently improved by 0.002.


**Smoothing**:
Nothing in the world is better than SG. It's fast enough and smooths everything. I even smoothed the predictions.

**Transit Zone Prediction**:
Nothing is more painful than getting 0 in submission results. The main problem is incorrectly predicted transit boundaries. In the test, they're wider or narrower than those in the training set. Using SG smoothing in the time direction to find boundaries. Ensuring light flux decreases at the left boundary and increases at the right.

**Features for Models**:
The feature construction process is similar to my baseline. We build a polynomial, find how to 'raise' the transit zone to align with the non-transit part. Used MAE and logpdf functions to evaluate how well we found the coefficient.
Build a general polynomial for the frequency sum. For each frequency, try to find coefficients pulling the transit as B and external parts as A to this polynomial. 1 - A/B is the target feature for individual frequencies. Sums across frequency intervals. I know from theory that some zones correspond to absorption areas of basic gases CH4, H2O, CO2, CO, NH3. But the best frequency areas for building features turned out to be different. I used a mix of genetic algorithm and manual selection to find the best intervals.

**Sigma Estimation**:
I only estimated average sigma per spectrum. Build regression from the difference between maximum and minimum predictions for the planet.

**Models**:
Two types: heuristic and CNN.

**Heuristic**:
Sum of features with weights that look realistic. 0.666 on training set and 0.666 on public. Two different approaches for predictions with differences <0.00018 and larger. For larger spectra, found coefficients can be used. For small ones, better not to.

**CNN**:
Main challenge was avoiding overfitting. Experimentally established maximum frequency count in window - 21. There's physical sense in this, close to distance between different gases' transmission bands. Important that model doesn't overfit on two bands. Two-headed model, one predicting 283 spectrum points, second head predicting average sigma. Loss function - GaussianLoss.
0.678 on training set, 0.684 on test.

**Mixing**:
Weight selection by best RMSE on training set.
0.681 on training, 0.692 on public.

**Didn't work on public**:
Predicting gas compositions and summing predicted TauREX gas spectra.

Final notebook https://www.kaggle.com/code/sergeifironov/new-blend-pdf?scriptVersionId=204505791
