# 1st Place Solution

Competition: ariel-data-challenge-2024
Rank: #1
Source: https://www.kaggle.com/c/ariel-data-challenge-2024/discussion/544317

[link to notebook](https://www.kaggle.com/code/cnumber/neurips-ariel-data-challenge-2024-final-submission)

# Introduction

We ( @daiwakun and @cnumber) would like to thank the organizers for hosting this fascinating competition, especially @gordonyip and @lorenzomugnai, who worked tirelessly so that competitors could focus on the more "interesting" parts.

Also, a huge applause to @jeroencottaar for maintaining first place for almost the entire duration of the competition. Although we managed to grab victory at the end of the race, we are confident that we would have lost had the competition period been one day shorter or one day longer.

Many ideas in our solution were found during a deep examination of the ExoSim2 and TauREx3 code used for data generation, including gain drift fitting and foreground processing. Although we didn't (or couldn't) find any leaks, we have to admit that our solution somewhat "hacks" the simulator. Nevertheless, we hope the hosts can make use of some aspects of our solution, and hopefully, it would be useful for the actual data.

For those who are curious about our significant score improvement during the last two days of the competition, we have decided to present what was going on in our team in [another post](https://www.kaggle.com/competitions/ariel-data-challenge-2024/discussion/544316)

# Solution

## Signal Preprocessing

Only the AIRS-CH0 channel was used because our solution heavily relies on the correlation between adjacent wavelengths, and also because we were unsure how to effectively utilize the FGS1 channel data due to issues with its distorted and fluctuating point spread function.

The public notebook (we wanted to put a link, but we couldn't find the notebook we used) was used with the following changes:

1. **Disabling Hot Pixel Processing**: This led to a significant jump on the leaderboard. We assume that the hot pixel processing leads to unwanted information loss of the pixels in the middle of the sensor, making it almost impossible to correct the noise introduced by the time-dependent PSF distortion (see image below). Perhaps it is the sigma clip algorithm that is doing the wrong thing and that there exist algorithms that can handle hot pixels properly, but we weren't able to find them.

2. **Foreground Processing**: Some teams have noticed that multiplying the final spectrum by a coefficient around 1.006~1.008 gives a huge boost. This is because a foreground is added to the signal during the ExoSim2 simulation [link to GitHub]. The correct way to handle this is to estimate the wavelength-dependent foreground signal and to subtract it from the signal in the central region. In our solution, we decided to estimate the foreground signal with the regions `[0:8]` and `[24:32]`, and subtracted it from the central region `[8:24]`. The effect of considering the foreground properly can be seen in [this discussion post](https://www.kaggle.com/competitions/ariel-data-challenge-2024/discussion/543853#3034987).





## Initial Dip Estimation for Each Wavelength

### Identifying the Transit Interval

For identifying the time location of the transit, the wavelength-averaged signal was used. After estimating the approximate time position using a rule-based algorithm, the exact time was identified with a fitting algorithm.

### Gain Drift

Gain drift refers to the variation in the detector's gain over time and across different wavelengths. These drifts can introduce systematic errors in the observed signal and must be corrected to accurately estimate the transit dip.

ExoSim2 models the gain drift as:

$$
(1 + f(t) \cdot g(\lambda))
$$

where \\( f(t) \\) and \\( g(\lambda) \\) are polynomial functions.

It is notable that the final function form is different from a two-variable polynomial \\( h(t, \lambda) \\), because while the wavelength and time components can be separated in the former, they cannot in the latter.

By using this functional form directly in the fitting described below, we were able to achieve high performance in the dip estimation.

### Dip Estimation with Gain Drift Fitting

The function used for fitting is as follows:

$$
y_{\text{pred}} = I(\lambda) \times \text{Box}(\lambda) \times (1 + f(t) \cdot g(\lambda))
$$

where \\( I(\lambda) \\) is the spectrum of the star, and \\( \text{Box}(\lambda) \\) describes the dip; it is 1 outside the transit and \\( 1 - d_\lambda \\) inside the transit.

The number of fitting parameters are as follows:

- \\( f(t) \\): 5 parameters
- \\( g(\lambda) \\): 5 parameters

It is worth mentioning that the optimal \\( I(\lambda) \\) and the optimal dip used in \\( \text{Box}(\lambda) \\), which minimize the mean squared error, can be found analytically if the other parameters are fixed, and they don't need to be considered by the fitting algorithm.

To stabilize the fitting process, a two-stage fitting was used. In the first stage, \\( I(\lambda) \\) was directly estimated from the raw signal with temporal averaging and fixed during the fitting. In the second stage, \\( I(\lambda) \\) was not fixed and was set optimally for each step during the fitting, while for the fitting parameters, the result from the first stage was used as the initial parameter. The error for each data point used in the fitting was estimated from the variation of the signal at each wavelength.

### Dip Error Estimation with Bootstrapping

The errors in the dip estimation at each wavelength differ due to the varying signal-to-noise ratios associated with each wavelength. [Bootstrapping](https://en.wikipedia.org/wiki/Bootstrapping_(statistics)) was used to overcome this problem and to estimate the dip error of each wavelength.

Further details are available in our code.

## Dip Estimation Considering Wavelength Correlations

Three models were used: Gaussian Process Regression, AutoEncoder, and Non-negative Matrix Factorization (NMF). They were ensembled with the ratio 6:2:2.

### Gaussian Process Regression

Nothing fancy, unlike the second-place solution.

A simple kernel composed of RBF and Matern kernels was employed, and the errors calculated by bootstrapping were passed to `sklearn.gaussian_process.GaussianProcessRegressor` so that the model can consider the uncertainty of each data point.

### AutoEncoder

We applied an autoencoder to capture relationships in the data.

Unlike PCA, which captures linear relationships, autoencoders can model more complex and nonlinear patterns.

Also, we expected that by training the model with MSE loss, the autoencoder model could take noise into account and recover the "optimal" spectrum.

Important points were to normalize the data for each exoplanet and to take the moving median of the dip spectrum to smooth the input dip spectrum.

The model we used is as below, with 4 nodes in the hidden layer:

```python
input_data = Input(shape=(input_dim,))
encoded = Dense(encoding_dim, activation='relu')(input_data)
decoded = Dense(input_dim, activation='linear')(encoded)

autoencoder = Model(input_data, decoded)
```

### NMF

Very similar to the autoencoder, but we added it to enhance the diversity.

Unlike with the autoencoder, the number of ranks was set to 5 for NMF.

### Spectral Components Identified by NMF

The plot below shows the identified spectral components by NMF for the training data.

It can be seen that the main contributing gases of each component are:

- **Component 1**: CO₂
- **Component 2**: CH₄
- **Component 3**: H₂O

This signifies NMF's ability to consider the correlation of the spectrum unsupervised.



### Sigma

We took the weighted average of the following components:

1. **Constant value** (planet and wavelength independent)
2. **Standard deviation of the smoothed predicted dip spectrum** (planet dependent, wavelength independent)
3. **Uncertainty predicted by Gaussian Process Regression** (planet and wavelength dependent)

The constant value played an important role in cases where the dip spectrum was almost constant but had some bias.

## The Contribution of Each Component

Since our solution incorporates a variety of ideas, we examined the contributions for those that seem important with late submissions.

| Method                                                           | Public      | Private     | Public Loss      | Private Loss     |
|------------------------------------------------------------------|-------------|-------------|------------------|------------------|
| **Final Submission**                                             | 0.7330321   | 0.7420624   | 0.0000000        | 0.0000000        |
| Only Gaussian Process Regression                                 | 0.7221480   | 0.7343485   | -0.0108841       | -0.0077139       |
| Only AutoEncoder                                                 | 0.7078056   | 0.7181137   | -0.0252265       | -0.0239487       |
| Only NMF                                                         | 0.7017943   | 0.7122631   | -0.0312378       | -0.0297993       |
| With Hot Pixel Processing                                        | 0.7021653   | 0.7224989   | -0.0308668       | -0.0195635       |
| Without Foreground Processing<br>with *1.008 to prediction       | 0.7225193   | 0.7298121   | -0.0105128       | -0.0122503       |

## What Didn't Work

### Absorption Spectra of Known Gases

We tried to use TauREx3 to fit the dip spectrum.

Although it worked really well on the training data, the leaderboard score was horrible, presumably because of the shift in the composition of the atmosphere.

We did try to identify the gases in the test data, but without any success.

### Denoising the Input Data with Machine Learning

It is quite hard to assume that taking the sum of the `[8:24]` channels is optimal.

In some cases, the optimal range could be `[9:23]` or `[10:22]`, or even taking a weighted sum could be optimal.

To resolve this problem, we tried several machine learning methods but didn't manage to beat `[8:24]`, probably due to the temporal spatial fluctuation of the signal.

## What We Wanted to Do If We Had More Time

- Utilize the FGS1 channel.

# Final Comments

Deep analysis of the simulation code led to critical ideas needed for our victory. The function for the gain drift or the foreground processing method couldn't have been found without observing the ExoSim2 code. Though we had believed that this is the destiny of competitions that use simulation-generated data, we were very surprised that some of the top teams were able to achieve high scores without using them, so again, a big applause to them.
