# 37th Solution

Competition: ariel-data-challenge-2025
Rank: #37
Source: https://www.kaggle.com/c/ariel-data-challenge-2025/writeups/37th-solution

## Introduction

We ( @ihiratch, @prgckwb, @itachicom615, @nashisandesu) would like to express our gratitude to the competition organizers.  It was a really fun and insightful challenge, and we look forward to participating again. 
<br>
Our approach consists of the following steps:
<ul>
<li>Preprocessing</li>
<li>Transit phase detection</li>
<li>Transit depth estimation</li>
<li>Sigma estimation</li>
<li>Postprocessing</li>
<li>Ensembling</li>
</ul>
We mainly used mathematical modeling instead of machine learning approach.  Our predictions focused on the average transit depth and sigma values.  We did not use wavelength-dependent predictions for either transit depth or sigma.  

## Preprocessing

For preprocessing, we relied on this excellent public notebook
( https://www.kaggle.com/code/vitalykudelya/neurips-non-ml-transit-curve-fitting ). We are very grateful to @vitalykudelya for sharing it.
The main modification we made was accelerating the linearity correction step with Numba.  This reduced the preprocessing time from about 6 hours to roughly 2 hours.  

## Transit Phase Detection

We detected the transit phase using the white light curve (i.e., the light curve averaged over all wavelengths).  

### Method Overview
We approximate the ingress and egress regions with straight lines.  The boundaries (ingress start, ingress end, egress start, egress end) can be estimated at the points where the white light curve begins to deviate from these lines (see the figure below).




### Details

- We first obtain the indices corresponding to the maximum and minimum values of differences of the white light curve.  
	- The minimum difference roughly corresponds to the middle of the ingress transition.  
	- The maximum difference roughly corresponds to the middle of the egress transition.  

- Around these indices, we construct local linear fits (“local lines”).  
- We then calculate the residuals between the local lines and the white light curve.  
- The points where the residuals started to deviate from zero are taken as the transit phase boundaries.  We detect a deviation when the residual exceeds `std(white light curve) × THRESH_RATIO` (a hyperparameter). 

## Transit Depth Estimation

We predicted the average transit depth for each planet using a mathematical model. The model was a simplified version of last year’s first-place mathematical model. ( https://www.kaggle.com/competitions/ariel-data-challenge-2024/writeups/c-number-daiwakun-1st-place-solution ) Many thanks to @cnumber and @daiwakun for sharing their excellent solution!

The idea is to lift the in-transit (IT) flux so that it smoothly connects with the out-of-transit (OOT) flux through fitting.  The prediction is made for the **average transit depth per planet**.  We did not predict wavelength-dependent transit depths (we tried, but performance degraded).  
<br>
Model: 

$$
y(t) = s \,(1 + d(t)), \quad t \in \text{OOT}
$$
$$
y(t) = r \cdot s \,(1 + d(t)), \quad t \in \text{IT}
$$

Optimization:

$$
\min_{s,\,r,\,d} \;\;
\sum_{t \in \mathrm{IT}} \frac{1}{n_{\mathrm{it}}}\,\bigl(y_{\text{true}}(t)-y(t)\bigr)^2
\;+\;
\sum_{t \in \mathrm{OOT}} \frac{1}{n_{\mathrm{oot}}}\,\bigl(y_{\text{true}}(t)-y(t)\bigr)^2
\quad
\text{s.t.} \quad 
\frac{1}{\,n_{\mathrm{it}}+n_{\mathrm{oot}}\,}\sum_{t} d(t) = 0
$$

Here:  
<ul>
<li><b>s</b>: the average flux level of each planet</li>
<li><b>d(t)</b>: the sensor drift term, modeled as a cubic polynomial</li>
<li><b>r</b>: the relative drop in flux during the transit phase</li>
<li><b>y_true</b>: the ground-truth white light curve</li>
<li><b>n_it</b>, <b>n_oot</b>: the number of samples in the in-transit (IT) and out-of-transit (OOT) phases, respectively</li>
</ul>

The constraint enforces that the average of the drift term is zero, ensuring identifiability with respect to *s*.  We solve this optimization independently for each planet.  The predicted transit depth for each planet is then computed as **1 − r**.  

<br>
For planets affected by **limb darkening**, we did not use the above model and optimization.  Instead, we estimated the transit depth as:   `1 − y_true.min() / y_true.max()`.  We analyzed the light curves and transit depths of planets with limb darkening and concluded that this approximation was not a bad estimation (see the figure below).





## Sigma Estimation

We estimated the average sigma for each planet from the relative standard deviations of the light curve in the time and wavelength directions.  We did not predict wavelength-dependent sigmas; instead, only the average sigma was predicted.  

From the light curve, we compute sigma_t (time-direction relative standard deviation) and sigma_w (wavelength-direction relative standard deviation), and then calculate the combined uncertainty for each planet as:

$$
\sigma_{\mathrm{combined}}
= \sqrt{\,w_t \,\sigma_t^2 + w_w \,\sigma_w^2\,}
$$

where w_t and w_w are hyperparameters.  We then define k, representing the relative uncertainty of each planet, as:

$$
k = \frac{\sigma_{\mathrm{combined}}}{\mathrm{Med}}
$$
Here, Med is the median of sigma_combined across all planets.  The predicted sigma for each planet is then given by:
$$
\sigma_{\mathrm{pred}} = k \cdot \sigma_{\mathrm{base}}
$$
where sigma_base is a hyperparameter representing the baseline sigma.  This ensures that planets with larger overall uncertainty are assigned larger sigma values, while planets with smaller uncertainty receive smaller values.  

The estimation methods for sigma_t and sigma_w are described below.  
### Estimation of sigma_t

For this part, we referred to the excellent public notebook:  
(https://www.kaggle.com/code/antonsibilev/very-fast-with-hot-pixels-enabled)  We sincerely thank @antonsibilev for sharing it.

The time-direction relative standard deviation sigma_t is computed as:

$$
\sigma_{t}
= \frac{1}{\bar y^{\text{oot}}}\,
\sqrt{ \frac{\operatorname{Var}(y^{\text{oot}})}{n_{\text{oot}}}
     + \frac{\operatorname{Var}(y^{\text{it}})}{n_{\text{it}}} }
$$

where `y_bar_oot` is the mean of the out-of-transit (OOT) white light curve. The sigma_t is calculated for each planet. The sigma_t can be interpreted as a measure of the signal’s intrinsic noise.
### Estimation of sigma_w

For each wavelength channel c = 1,…,283, we compute the in-transit and out-of-transit mean flux as:

$$
\mu_{\mathrm{it},c} = \frac{1}{n_{\mathrm{it}}}\sum_{t=1}^{n_{\mathrm{it}}} y_{t,c}
$$
$$
\mu_{\mathrm{oot},c} = \frac{1}{n_{\mathrm{oot}}}\sum_{t=1}^{n_{\mathrm{oot}}} y_{t,c}
$$


The rough estimator of the transit depth at wavelength c is then:

$$
\delta_c = 1 - \frac{\mu_{\mathrm{it},c}}{\mu_{\mathrm{oot},c}}
$$

After smoothing delta_c with a Savitzky–Golay filter, the wavelength-direction relative standard deviation sigma_w is computed as:

$$
\sigma_w = \mathrm{STD}(\delta_c)
$$

This allows us to estimate the variability of transit depths across wavelengths.


## Postprocessing

For planets with anomalous light curve shapes or where transit phase detection failed, we applied a fallback procedure.  In these cases, the transit depth was estimated as:   `1 − y_true.min() / y_true.max()`. Additionally, we multiplied the predicted sigma by about three to prevent performance degradation.


## Ensembling

We prepared 100 different hyperparameter configurations and ensembled their predictions.  The hyperparameter configurations were explored using Optuna with the TPE sampler.  We split the data into train and validation sets, and searched hyperparameters on the train set, and then selected candidate configurations that achieved good scores on both train and validation.  

Ensembling the 100 predictions slightly improved the local score (from 0.437 to 0.440), but did not provide any gain on the Public or Private leaderboard.  

## What Did Not Work

Predicting wavelength-dependent transit depths did not work well.  Using only the average transit depth consistently yielded better performance.
