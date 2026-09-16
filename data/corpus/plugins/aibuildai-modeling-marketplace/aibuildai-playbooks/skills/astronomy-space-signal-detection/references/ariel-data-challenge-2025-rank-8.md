# 8th place solution

Competition: ariel-data-challenge-2025
Rank: #8
Source: https://www.kaggle.com/c/ariel-data-challenge-2025/writeups/8th-place-solution

I would like to express my gratitude to Kaggle and University College London for hosting this competition. I learned a great deal, and it was truly an exciting experience.

</br>

# Summary
1. Preprocess : noise reduction
2. Stage 1 :  transit time prediction & filtering bad samples
3. Stage2 : physics-based transit modeling & 2d global fitting
4. Postprocess : ml-based sigma prediction & simple refinements

</br>

# Preprocess : noise reduction
- Applied all standard preprocessing functions except for the mask_hot_dead function.
- For time binning, 30 is used for airs_ch0 channels, and 30 × 12 is used for fgs1 channel.
- The spatial dimension is cropped to the range 8 ~ 24.
- The remove_outlier function modifies outliers in the data using the local mean.
- The effect of the remove_outlier function can be confirmed in the following figure.

```python
def remove_outlier(data, window = 100, threshold = 5.0):
    df = pd.DataFrame(data)
    
    mean = df.rolling(window = window, center = True, min_periods = 1).mean()
    std = df.rolling(window = window, center = True, min_periods = 1).std()
    
    mask = (df - mean).abs() > (threshold * std)
    
    df[mask] = mean[mask]
    data = df.to_numpy()
    return data
```



</br>

# Stage1 : transit time prediction & filtering bad samples
- The input is obtained by averaging all channels and then normalizing.
- Transit and system are modeled with simple polynomials to predict T1, T2, T3, T4.
- Additionally, Success or not is predicted to prevent negative scores from bad samples.
- If success == False, wl and sigma are set to the train dataset’s mean and std.
- Criteria for success = False:
    1. Fitting error > 1e-3 
    2. T1 < 0.01 or T4 > 0.99
- The Stage 1 fitting results is visualized in the following figure.



</br>

# Stage2 : physics-based transit modeling & 2d global fitting
- Implemented a custom nonlinear limb darkening pipeline (c1, c2, c3, c4), inspired by the batman library.
- To avoid unrealistic cases (where c1 + c2 + c3 + c4 > 1), each coefficient is capped at 0.25.
- Used least_squares for global fitting across all channels with parameters: rp, c1, c2, c3, c4, P, sma, i, t0.
- Applied tanh to c1–c4 (range -1 to 1) for better convergence stability and easier max constraints.
- Constructed jac_sparsity to predefine parameter correlations and speed up convergence.
- Adopted the previous competition’s 1st-place approach: system = 1 + f(time) × g(wavelength).
- To mitigate degeneracy between limb darkening effect and system effect, a 2-step fitting is used:
    1. Step 1 : f & g modeled as 3rd-order polynomials.
    2. Step 2 : f & g modeled as 4th-order polynomials.
- This allows limb darkening coefficients to converge first, reducing errors from degeneracy.
- For faster computation, wavelength binning of 4 is applied.
- Since channels beyond 200 in airs_ch0 have severe noise, a window size of 50 is used.
- The target of the Stage 2 fitting is visualized in the following figure.

```python
with Pool(processes = os.cpu_count()) as pool:
    res = least_squares(
        fun = fun,
        x0 = x0,
        bounds = bounds,
        method = 'trf',
        jac_sparsity = jac_sparsity,
        args = (time, star_info, param_info, targets),
        workers = pool.map,
        verbose = 2 if plot else 0,
    )
```





</br>

# Postprocess : ml-based sigma prediction & simple refinements
- inputs = parameters obtained from stage 1 and stage 2 for the training data.
- targets =  optimal sigma per sample, computed over np.linspace(5e-5, 2e-3, 500).
- Applied a 4-fold ensemble of GradientBoostingRegressor for inference.
- For the fgs1 channel, results are simply scaled by ×2 from airs_ch0 channels.
- Additional refinements -> wavelength scaling, gaussian_filter1d, and PCA.

```python
inputs = np.concatenate([
    T,
    T[:, 3:4] - T[:, 0:1],
    T[:, 2:3] - T[:, 1:2],

    rp[:, :-1].mean(1, keepdims = True),
    c1[:, :-1].mean(1, keepdims = True),
    c2[:, :-1].mean(1, keepdims = True),
    c3[:, :-1].mean(1, keepdims = True),
    c4[:, :-1].mean(1, keepdims = True),

    rp[:, :-1].std(1, keepdims = True),
    c1[:, :-1].std(1, keepdims = True),
    c2[:, :-1].std(1, keepdims = True),
    c3[:, :-1].std(1, keepdims = True),
    c4[:, :-1].std(1, keepdims = True),

    cost,
    nfev,

    star_info.values,
], axis = 1)
```

</br>

# Code
https://github.com/siwooyong/NeurIPS-Ariel-Data-Challenge-2025
