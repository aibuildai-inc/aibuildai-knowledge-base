# 33rd place solution

Competition: ariel-data-challenge-2025
Rank: #33
Source: https://www.kaggle.com/c/ariel-data-challenge-2025/writeups/33rd-place-solution

**Data exploration**

[This notebook](https://www.kaggle.com/code/ddimonte/understanding-the-data) provides detailed exploration of the data format and applies processing steps borrowed from the [binning and processing notebook](https://www.kaggle.com/code/gordonyip/update-calibrating-and-binning-astronomical-data). Key changes in this notebook include minimal time binning (just enough to standardize AIRS-CH0 and FGS1 lengths), updated masking behavior, the addition of inpainting for missing data, and extensive before/after visualizations.

In [this notebook](https://www.kaggle.com/code/ddimonte/process-data) we fit a polynomial baseline to the out-of-transit regions of each light curve, using smoothing and change point detection to identify the transit onset and offset. However, we found that some transits in our dataset are incomplete or contain somewhat complex background changes, making baseline normalization unreliable for all cases without more careful consideration. To address this, we briefly attempted to fit a BATMAN transit model to the data, but ultimately decided not to pursue this further due to time constraints and the need for robust validation.

We then investigated the ratio between the provided ground truth transit depth and the maximum observed transit depth from the light curve. We found that the ground truth value is not always at the minimum of the light curve or a fixed fraction of the fitted model, and can even be lower than the observed minimum for some planets. This suggests that other parameters besides for the visible transit depth in the light curve alone play a significant role in determining the true transit depth such as planet and star data or the shape of the transit curve. This is likely due to transit characteristics and limb darkening.

[Light curve examples including group truth]

*Figure 1:* Light curve examples including ground truth estimate.

**Preprocessing** 

[This notebook](https://www.kaggle.com/code/ddimonte/submission-code) contains the final submission code with all the necessary preprocessing on the test data. Much of the code was taken from the [binning and processing notebook](https://www.kaggle.com/code/gordonyip/update-calibrating-and-binning-astronomical-data). We also use code from the ruptures library.
Preprocessing steps:
- Load and Calibrate Detector Signals
- Load Calibration Files and Apply Detector Cleaning
    - Mask hot/dead pixels: Replace unusable data using mask maps and dark files.
    - Non-linearity correction
    - Flat Field Correction: Correct for pixel-to-pixel sensitivity variations using loaded flat field maps.
    - Dark subtraction: Subtract the appropriate dark current background.
    - Correlated Double Sampling
- Time binning 12 to FGS1 to align time dimensions.
- Inpainting: Fill remaining missing or masked regions using biharmonic inpainting treating time as channels.
- Spatial Summation: Sum along spatial axes to produce light curves and prepare input shapes appropriate for the model.
- Median filtering kernel size 101
- Downsampling stride 10

**Model**

The overall model architecture is shown in Figure 2. The preprocessed data [wavelength x time] is normalized per wavelength and passed through a Time-Reducing Residual Stack. The resulting feature map is pooled and then concatenated with the wavelength means z-scores, the wavelength standard deviations z-scores, and the normalized planet features. These scores are calculated using summary statistics from the entire dataset in order to incorporate information about the sample's placement in the population distribution into the model after our earlier observations shown in Figure 1. The combined vector is passed through a fully connected layer, whose output feeds three separate fully connected output heads. The first two heads are used to compute the per-wavelength prediction, and the third head is used for the uncertainty (sigma) values. For the prediction, we calculate the relative change between the first two outputs. For the third output, we train the model to output log(sigma) so we take the exponential of the third output head for our final sigma value.

The Time-Reducing Residual Block details are shown in Figure 3. In the figure N represents the block number from 1 to 6 since 6 blocks are stacked in the final model. The wavelength dimension uses circular padding and increasing kernel dilation 3<sup>N-1</sup> so that all of the wavelengths' features are convolved with each other within 6 stacked residual blocks. The time dimension uses 0 padding and a kernel dilation of 1 so the time field of view remains within the nearest times from small increments to large increments as the time dimension is pooled in successive blocks. In our model we used a time dimension pooling kernel of 4 for the first 4 blocks and then eased off down to 2 to maintain the desired data size for a 6 block stack and to prioritize faster compression of the time dimension earlier.

We trained the model with a combined loss utilizing both MSE as well as the GLL error used for scoring.

[Model flowchart]

*Figure 2:* Model flowchart. The input is the pre-processed data array x, and the outputs are the predictions for the requested wavelengths and the associated sigma values.

[Detailed view of the Time Reducing Residual Block N for N as 1 through 6]

*Figure 3:* Detailed view of the Time Reducing Residual Block N for N as 1 through 6.

**Augmentation**

To improve model generalization and robustness and decrease overfitting, we implemented several data augmentation techniques during training:

- Time Flipping: Each light curve was flipped along the time axis to expose the model to both forward and reversed transit scenarios.

- Variable Downsampling and Offsets: We used strides of 8, 9, or 10 and multiple random offsets, simulating the effect of time running at different speeds and producing light curves with varying sampling densities. This approach also increased the representation of incomplete transits, helping the model learn to handle edge cases and irregular data spans. Because the signals were median-filtered (kernel size 101), the incremental effect of offset variation is modest, but still introduces some diversity.

- Additive Linear Trend: For each wavelength channel, we injected a random linear signal with a maximum slope set equal to the channel’s own range. The maximum was set per wavelength, but the randomization was done per planet observation.

- Additive Sinusoidal Trend: For each wavelength channel we injected a random sinusoidal signal with low frequency.

- Planet Parameter Noise: Small, zero-mean Gaussian noise (σ = 0.1) was added to each set of normalized planet parameters fed into the model. This reduces overfitting and allows the network to be more robust to small errors or uncertainty in planet/star parameters.

**Ensembling methods**

To aggregate the model predictions for each planet, we explored several ensembling approaches seen in our [submission notebook](https://www.kaggle.com/code/ddimonte/submission-code). We found that the basic average provided the best results due to how we were predicting our values and sigmas. Each planet had predictions from (potentially) multiple observations, 5 different offsets of stride 10, and multiple model predictions. Methods we tried:

- Basic Average: A simple average of all predictions for each wavelength channel for each planet, both for the value and the sigma.
- Best Sigma Selection: Selecting only the single set of predictions (per planet) with the lowest mean predicted uncertainty across all wavelengths.
- Best N Average: A simple average of the N rows of data per planet that had the lowest mean sigma values.
- Weighted Averaging: Weighted each prediction by the inverse of its predicted uncertainty, computing weighted means and associated uncertainties for each wavelength.

**Other notes:**

*Stratification*

We explored stratification based on star and planet features [[notebook](https://www.kaggle.com/code/ddimonte/stratification)], hoping to create folds that better represent different subclasses or difficulty levels. However, the lack of strong natural clusters, weak correlations between features and prediction error, and overlapping distributions showed that meaningful strata did not exist in the data for the features we analyzed. As a result, simple random folds were used.

*Other models*

**Citations**

- C. Truong, L. Oudre, N. Vayatis. Selective review of offline change point detection methods. Signal Processing, 167:107299, 2020.
