# 5th place solution

Competition: ariel-data-challenge-2025
Rank: #5
Source: https://www.kaggle.com/c/ariel-data-challenge-2025/writeups/5th-place-solution

This competition was a great opportunity to take a break from deep learning and do some modeling. Huge thanks to Kaggle, @gordonyip and all the other organizers, looking forward to the 2026 edition!

At the base of my solution is a physical model with parameters fit to match the noisy data. Next, I apply some post-processing to refine the results, most importantly gradient boosting.

<hr>

# Pipeline
1. Calibrate the data, remove outliers, do 5x time binning (T=1125) and 6x (W=47) spectrum binning;
2. Fit the physical model with regularized Gauss-Newton algorithm;
3. Refine the outputs with PCA and Boosting.

<hr>

# Physical Model
The model is similar to what was proposed in Ariel 2024. However, transit depth is much more complex now.



**1. Star Spectrum** is calculated as `sensor_data.mean(axis=0) / (transit_depth * poly_drift).mean(axis=0)`. No parameters to optimize specifically for star spectrum.

**2. Transit Depth** is the main part of the model. It takes in a bunch of parameters and outputs how much light remains for each wavelength at each moment of time. Basically, it is the same model as described in @junkoda [notebook](https://www.kaggle.com/code/junkoda/limb-darkening): 

$$\textbf{transit depth}(t, w)= 1 - \int_{D_{tw}} F(x)dx.$$

Here D is an intersection of stellar and planetary discs at each time moment t at each wavelength w. F is the star intensity according to the limb darkening law. I used nonlinear law with six coefficients covering x^0.5 to x^3. The numerical integration was also taken from @junkoda`s notebook (it *was* necessary after all :D ). Here is the full parameter list:
- **Rp mean** — scalar;
- **Rp variation** — 15 values linearly interpolated to W values;
- Impact factor **b** — scalar in [0, 1]. Actually can be larger than 1 but it didn't help anyway;
- **limb coeffs** — six coefficients for nonlinear limb darkening model;
- **ingress**, **egress**  — two scalars for start and end of the transit window.

I also experimented with adding **orbit radius** to take trajectory curvature into account. It produced better results when testing on samples manually generated with batman, but didn't work well on the training data.

Notably, the transit dip is smooth and has length varying over spectrum. A planet with the biggest Rp range illustrates this well:



**3. Polynomial Drift** is 1 + f(t, w) where f is a bivariate polynomial with degree 4 in time dimension and degree 2 in wavelength dimension, resulting in 15 parameters.

<hr>

# Fitting and Implementation

**Optimization**. The output of the model is differentiable w.r.t. all the parameters described above. I assume the noise to be gaussian and use MSE loss with wavelengths weighted according to noise deviations. I ended up using Levenberg-Marquardt method for fitting (and should thank @jeroencottaar whose solution of GWI made me do some digging on optimization).

Optimization runs for 220 LM iterations and mainly fits all parameters simultaneously in one stage. The only exceptions are that polynomial drift is locked before 70th iteration and Rp variation is locked before 180th iteration.



**Sigma prediction** is constant over spectrum and is calculated simply as `mu.std() * 1.6`.

**Cut transits**. Since ingress / egress were fitted automatically, I did not have to add any special treatment for planets with one edge. 

**Performance**. Fitting time for one planet depends on the transit width and usually takes around 20-25 seconds on P100 (FP64). All logic was implemented in PyTorch with autograd. The transit depth integral was estimated with 40 rings.

**AIRS / FGS channels**. Applying the model to AIRS only (and setting FGS mu as mean AIRS mu) would yield ~0.542 on Public LB. Fitting FGS too  improved the score to 0.546.

<hr>

# Refining the outputs

**Linear trend and instability**. Somehow my method fits Rp with additional linear trend and unstable values at the right edge of the spectrum. I haven't found a better way than to simply remove this trend as a post-processing step and set `Rp[-50:] = Rp[-50]`. Here are the raw outputs for the very first planet:



**Planets with high impact factor**. For planets with b > 0.75 my model fitted mu to values ~4-6% larger than ground truth. This is another mystery I failed to solve. I added a special processing for these planets:
```
if model.get_b() > 0.75:
    mu = mu / 1.05
    sigma = sigma * 7.0
``` 

**Applying PCA** on predicted Rp variation improved Public LB score by 0.007. I used 3 components.

**Gradient Boosting** worked particularly good for me and improved my score by 0.028 during the last week. I used it to refine mu and sigma for both AIRS and FGS:
- Input features are `star_info.csv` data and all fitted parameters;
- Outputs are constant shifts (for example, `mu_airs` becomes `mu_airs + dmu_airs` where `dmu_airs` is scalar boosting output).

<hr>

# Scores

The first row shows the score with trend and unstable values already removed.

| Submission | Public LB | Private LB |
| --- | --- | --- |
| Fit model on AIRS only | 0.542 | 0.550 |
| Fit model on FGS too | 0.546 | 0.554 |
| Add PCA | 0.553 | 0.557 |
| Add AIRS mu & sigma boosting | 0.577 | 0.586 |
| Add FGS mu & sigma boosting | 0.581 | 0.589 |

# Code 

GitHub: [github.com/Alehandreus/ariel-2025](https://github.com/Alehandreus/ariel-2025)
Kaggle notebook: [kaggle.com/code/alehandreus/ariel-2025-inference](https://www.kaggle.com/code/alehandreus/ariel-2025-inference)
