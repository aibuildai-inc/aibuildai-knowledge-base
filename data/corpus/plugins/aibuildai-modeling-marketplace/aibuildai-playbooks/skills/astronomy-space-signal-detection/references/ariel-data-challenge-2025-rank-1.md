# 1st place solution: Bayesian Inference, of course

Competition: ariel-data-challenge-2025
Rank: #1
Source: https://www.kaggle.com/c/ariel-data-challenge-2025/writeups/1st-place-solution-bayesian-inference-of-course

[Link to code to train and submit](https://www.kaggle.com/code/jeroencottaar/ariel2025-1st-place-bayesian-exoplanet-analysis)
This year I'm also releasing the full development history: [https://github.com/jcottaar/ariel2](https://github.com/jcottaar/ariel2) <br>
I'd like to start by thanking the organizers for another great competition. I had a rather frustrating start (having somehow managed to lose all my code from last year), but greatly enjoyed it overall. The Bayesian approach is not used nearly as often as it should be these days, and I'm glad to have the opportunity to show its value here.

My solution is quite similar to last year's; my [NeurIPS talk](https://neurips.cc/virtual/2024/107111) from the associated workshop can be a useful introduction. I'll briefly discuss the differences compared to last year in a reply to this post.
# 1. Introduction

Transit analysis is the most common method for detecting and studying exoplanets. A transit occurs when a planet crosses in front of its star relative to Earth, obscuring part of the starlight. By observing how much the light is reduced (the transit depth) as a function of wavelength, we can learn the properties of the planet. But with a tiny planet passing in front of a huge star, this problem has a very low signal to noise ratio. The challenge to us: find the transit depth, including confidence intervals, from synthetic raw spectroscopic signals as the Ariel satellite might see them in the future (For a more extensive overview, see https://www.kaggle.com/competitions/ariel-data-challenge-2025/overview).



This competition is at first glance a retread of a similar competition last year, but the data is far more realistic this time around. I think I figured out some of the changes, but not all of them - as we'll see later on.<br>
Note that there are two sensors in play, the single-wavelength FGS sensor and the spectroscopic AIRS sensor. In this writeup I'll focus mainly on the AIRS sensor.<br>
My general approach consists of three steps, as visualized below:
- **Preprocessing**: starting from the raw sensor data (expressed in counts), compute the amplitude signal *S(λ,t)* as a function of wavelength *λ* and time *t*.
- **Bayesian Inference**: from *S(λ,t)*,  compute the inferred transit depth *D(λ)* and an uncertainty margin on this *σ(λ)* . This step is done using a principled Bayesian approach.
- **Fudging**: throw the principles out the window, and apply various fudge factors to *D(λ)* and *σ(λ)* . These fudge factors are chosen to optimize score on the training data.



In sections 2, 3, and 4, I'll discuss each of these steps in turn. I'll finish with some closing thoughts in section 5 - mainly discussing why my approach isn't actually working...
# 2. Preprocessing

The raw data is in counts per pixel; we first need to apply several steps to come to a photon count per pixel. This is explained in [a notebook shared by the hosts](https://www.kaggle.com/code/gordonyip/calibrating-and-binning-ariel-data). I use a time binning of 5 frames for AIRS, and 50 frames for FGS (this puts them approximately in sync). Hot pixel invalidation is disabled. I also added a simple cosmic ray removal. <br>
The opportunity for some real improvement lies in the wavelength binning step, i.e. summing over the dispersion axis. The baseline is simple summing, but there's several effects in the data that make it possible to do better. Most notably, there's the infamous jitter, which we can see most clearly by doing a PCA (with some additional preprocessing) over all AIRS frames in the train data:



The jitter shapes presumably correspond to pointing error and defocus.

The jitter shapes sum to approximately zero over the dispersion axis; this is why simple summing works OK. It also means that any more advanced preprocessing scheme is forced to deal with the jitter, so there's a high barrier to entry. Still, there are several effects in the data that such a scheme can take advantage of:

- There are invalid pixels.
- The noise is not quite Poisson noise (simple summing minimized noise only for Poisson noise); the noise is lower for brighter pixels, and there are occasional very noisy pixels.
- The jitter shapes don't quite sum to zero, presumably because detector calibration is not perfect.
- There is a constant background signal that we'd like to remove.

I spent more than half of my time on this competition trying to set up a principled preprocessing flow optimally handling all these effects. Unfortunately, I haven't managed this yet - you'll find in the code several disjointed steps attempting to deal with the above. They also tend to work only in certain configurations and combinations, which I don't yet fully understand. It still leads to an improvement of ~0.01 over simple summing - I'll crack this one next year...
# 3. Bayesian Inference

Now that we have the photon counts as a function of wavelength and time, we are ready to infer the exoplanet transit depth using Bayesian Inference. <br>
**Bayesian Inference** (BI) is a powerful statistical approach, based around defining a *prior* (a statistical belief about reality) and *observations* (some form of new information). Using Bayes' law, we then combine these to find the *posterior* (an updated belief about reality). In our case, this means:
- **Prior**: a description of the physics that affect the final measured signal, describing for example detector noise, drift, and the transit behavior (including the transit depth itself) as formal distributions.
- **Observations**: the provided measurements.
- **Posterior**: a breakdown of the observations into the various elements defined in the prior (see figure below). From this we can simply read out the desired transit depth. Importantly, the posterior is not just a single point, it's a distribution. By taking samples from this distribution we can find the required confidence intervals (and even full covariance matrices, although that is not asked of us here).

There are two key elements to applying BI: defining the prior (section 3.1), and applying Bayes' law to do the inference (section 3.2).
## 3.1 Prior definition

The signal is decomposed into four key components, as shown visually here:



This table shows a detailed description of all prior elements:

| Prior element                 | Description                                                                                                                                                                                                                                                                                                         | Tuning and hyperparameters                                                                                                                     | Degrees of freedom                     |
| ----------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------- |
| Noise                         | Uncorrelated Gaussian per time and wavelength                                                                                                                                                                                                                                                                       | Standard deviations found in preprocessing                                                                                                     | 1350 (FGS) + 317250 (AIRS)             |
| Star spectrum                 | Uncorrelated value per wavelength                                                                                                                                                                                                                                                                                   | Not regularized (infinite sigma)                                                                                                               | 283                                    |
| Drift                         | Third order polynomial over time and wavelength                                                                                                                                                                                                                                                                     | Not regularized (infinite sigma)                                                                                                               | 3 (FGS) + 12 (AIRS)                    |
| Transit window                | Found using batman package, with the following free parameters:<br>- Transit mid-time t0 (separate for FGS and AIRS)<br>- Semi-major axis sma<br>- Period P<br>- Orbital inclination i<br>- Quadratic limb darkening u0 and u1 (including linear dependence on wavelength)<br>- Transit depth Rp^2/Rs^2 (see below) | Covariance matrix learned on training set using maximum likelihood estimation (MLE)                                                            | 11 (6 limb darkening, 2 t0, sma, P, i) |
| Transit depth: mean           | Single value                                                                                                                                                                                                                                                                                                        | Lightly regularized (sigma=0.01)                                                                                                               | 1                                      |
| Transit depth: variation      | Decomposed into three components below                                                                                                                                                                                                                                                                              | Scaling factor determined per planet using MLE (this is the only hyperparameter tuned during inference)                                        | N/A                                    |
| Transit depth: variation FGS  | Single Gaussian value                                                                                                                                                                                                                                                                                               | Standard deviation found on training set                                                                                                       | 1                                      |
| Transit depth: variation AIRS | Gaussian Process over wavelength, sum of two squared-exponential kernels                                                                                                                                                                                                                                            | Kernel hyperparameters found on training set using maximum likelihood estimation                                                               | 282                                    |
| Transit depth: variation PCA  | Fixed basis functions obtained from PCA analysis                                                                                                                                                                                                                                                                    | PCA shapes and variation magnitudes found on training data (this year there seems to be no train-test population shift for the transit depths) | 5                                      |
## 3.2 Solver

Having defined the prior, finding the posterior is 'just' a matter of applying Bayes' law. The prior is fully Gaussian, which means the inference should just be a single matrix operation. But as usual, there's a snag. In our case, it's that the relationship between the prior parameters and the observation is non-linear (because of the non-linear modeling in batman, and because some prior elements are multiplied rather than added). <br> 
To deal with this non-linearity, we linearize the model around the mean of the posterior. This is repeated iteratively (i.e. apply BI -> linearize around the new mean of the posterior -> repeat). Each step we also update the single hyperparameter (the scaling of the transit depth) using gradient descent. <br> 
This approach does mean we need a decent starting point, which is handled as follows:

- **Grid search**: sweep over a grid of transit depth and transit mid-time to find initial values. Other transit parameters are set to the given approximation per planet (with no limb darkening).
- **BFGS**: fit drift and all transit parameters, using BFGS as implemented in ```scipy.optimize.minimize```. In this step we only consider the mean over wavelengths of the AIRS signal. 
- **Bayesian Inference**: the full non-linear BI solver as described above. My winning submission uses 8 iterations of the solver; the code linked here only does 4 iterations to fit training and inference into the 9 hour submission limit (this only costs 0.001 in the score).

The prediction of the transit depth and the covariance matrix of its uncertainty can be read out directly in the posterior. The covariance matrix is approximated based on 200 samples; this is mainly a holdover from last year, when there were too many parameters to construct it explicitly.

There are two additional tricks that I apply if the residual of the first two steps (grid search->BFGS) is suspiciously high:
- Use transit parameters for a different planet as starting point.
- Try chopping off the first and last few frames of the signal (these sometimes have a high residual for reasons I don't understand).
  
# 4. Fudging

The prediction and uncertainty obtained above can still be improved further with some post-hoc calibration. Note that this is anathema to a proper Bayesian approach - more on that below.

We adapt the mean of the transit depth per sensor based on fitting:
 - A constant offset
 - A multiplicative factor
 - Dependence on the first limb darkening parameter

The variation of the transit depth over wavelength is not fudged, i.e. not changed in this step.

We adapt the uncertainty of the transit depth per sensor based on fitting:
- A constant offset
- A multiplicative factor for the mean of the transit depth
- A multiplicative factor for the variation of the transit depth over wavelength
- Dependence on the magnitude of the variation of the transit depth over wavelength

In total, the fudging has 12 parameters to fit. These are determined on the full training set, based on whichever values optimize the competition metric.
# 5. Final thoughts - we're missing something!

The fact that fudging helps the score is extremely worrying. And it's not subtle: last year, my solution without fudging could have gotten 1st place; my solution this year without fudging would have ended up around ~20th place. It means there's something missing in the prior, i.e. it does not describe reality (or rather the synthetic data generation process) accurately.

I've spent a lot of time trying to find this gap, but haven't gotten much closer. I've come to believe that the issue lies in the transit modeling, but none of the variation I tried (such as different limb darkening models) helped. I'm convinced that finding whatever's going on could boost our scores to well over 0.700.

Some more direct evidence of this issue can be found by comparing different transits for the same planet; the plot below shows the ratio of the transit profiles (with some preprocessing including low pass filtering). Nothing in the prior (and nothing else I can come up with) can explain this. See also [this discussion](https://www.kaggle.com/competitions/ariel-data-challenge-2025/discussion/602425) and [the code](https://www.kaggle.com/code/jeroencottaar/demonstrate-apparent-label-issue).


I hope the organizers are willing to reveal some clues about this, because I don't think we'll figure it out next year either without some help.

Finally, I'll wrap it up with an overview of how much various elements of my solution contribute to the score:

| Change to model                                                                                                            | Impact on private test score     |
| -------------------------------------------------------------------------------------------------------------------------- | -------------------------------- |
| Simplified loader (host's flow + cosmic ray removal), no jitter correction or background removal                           | -0.013                           |
| Don't remove first or last few frames of suspicious transits                                                               | -0.002                           |
| Don't use Gaussian Process for transit depth, just use PCA<br>Note: this variation does not involve any Gaussian Processes | -0.007                           |
| Don't use PCA for transit depth, just Gaussian Process                                                                     | -0.011                           |
| Disable all fudging - rely purely on the outcome of the Bayesian predictions                                               | -0.152<br>(Last year: -0.000...) |
| Replace all fudging by a single multiplicative factor for the uncertainty                                                  | -0.017                           |
| Disable fudging of sigma prediction based on AIRS variation                                                                | -0.005                           |
| Disable fudging of mean prediction based on limb darkening parameters                                                      | -0.005                           |
| Don't regularize the 11 transit parameters in prior                                                                        | -0.003                           |
