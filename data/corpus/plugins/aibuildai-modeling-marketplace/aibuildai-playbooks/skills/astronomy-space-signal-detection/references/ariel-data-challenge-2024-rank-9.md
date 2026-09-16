# 9th place solution

Competition: ariel-data-challenge-2024
Rank: #9
Source: https://www.kaggle.com/c/ariel-data-challenge-2024/discussion/543983

First of all, I want to thank the hosts and Kaggle team for this competition.

# Data Preprocessing
- Signal calibration is the same as the host's one, except binning and hot-pixels handling are omitted.
- Signal splitting is based on the first and second derivatives like in [3rd place solution](https://www.kaggle.com/competitions/ariel-data-challenge-2024/discussion/543944)

# Spectrum (μ and σ) Prediction
The prediction procedure can be divided into 3 parts:
- coarse estimation of μ for different chunks of wavelengths;
- σ estimation;
- μ refinement based on σ.

### Coarse Estimation of μ
- μ is estimated as the relative drop of the signal during transit. I used two approaches to calculate μ. The first one is a slightly modified version of the polynomial fit method proposed by @sergeifironov. The second approach is to take the difference between the signals around the fall region and divide it by the higher one. Since the signal can include a low-frequency trend, it is important to consider only small regions (used 90 timestamps). The final μ is the weighted combination of these two approaches.
- The estimate of mean μ_1 is calulated over the entire 282 wavelengths (1 big chunk). The estimate of μ_47 for individual wavelengths is obtained by dividing the wavelengths into 47 chunks (each contains 6 pixels).
- In both cases the signal is filtered by Butterworth over time. Moreover, in the case of 47 chunks, the signal is additionally filtered by Hann window (1D convolution) over wavelengths.

### Estimation of σ
- The approach is increadibly simple. Calculate μ_4 (4 μs for 4 chunks, each with ~70 pixels) and just take the std() over these 4 values: σ = μ_4.std().

### Refinement of μ
- Despite the filtering over wavelengths, μ_47 still has extreme deviation. Thus it is essential to filter the obtained μs again .
- The final prediction of μ is the weighted combination of the mean μ_1 (1 big chunk) and μ_47 (47 chunks). The important find is the greater the TRUE σ, the better μ_47 approximates the TRUE μ. So the weight w_47 of μ_47 grows monotonically with estimated σ.

In this discussion I have mentioned only the main details and omitted many small features.
The inference code: https://www.kaggle.com/code/arsenypoyda/ariel-inference-9th-place
