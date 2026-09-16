# 29th Discovery and Solution

Competition: ariel-data-challenge-2025
Rank: #29
Source: https://www.kaggle.com/c/ariel-data-challenge-2025/writeups/29th-discovery-and-solution

I would like to express my sincere gratitude to Kaggle and University College London for hosting this competition. I learned a great deal, and it was truly an exciting and rewarding experience. I would also like to thank my teammates @ajinomoto132 @chenzhenyuan @atamazian @larrylin666 —together, we proudly earned a silver medal in this competition.


# **Summary**

* **Preprocessing:** Noise reduction
* **Stage 1:** Transit time prediction and smoothing of noisy samples
* **Stage 2:** Transit modeling and end-to-end neural network
* **Postprocessing:** Simple refinements for the transit model and ML-based sigma prediction for the neural network


# **Data Preparation**
Based on the official submission baseline, we made the following modifications: we adopted a binning factor of 5 to preserve richer temporal information and applied background noise removal to improve signal quality.

## augmentation
We augment the dataset by pairing each signal with all available calibration files, rather than only its nominal pair, to increase sample diversity and improve model robustness.
```markdown
signal_0 + calibration_0 → sample1 
signal_0 + calibration_1 → sample2 
signal_1 + calibration_0 → sample3 
signal_1 + calibration_1 → sample4
```

## smoothing and normalization

```python
def smooth_data_lambda_batch(train_signal, win=3):
    """
    Smooth spectral data with Gaussian filter (batch version).
    
    Args:
        train_signal: numpy array of shape (batch_size, n_channels, n_wavelengths)
        win: window half-size (default=3)
    
    Returns:
        Smoothed signal of shape (batch_size, n_channels, ?)
    """
    batch_size, n_channels, n_wavelengths = train_signal.shape

    def gaussian_kernel(size=7, sigma=1.0):
        x = np.arange(-size//2 + 1, size//2 + 1)
        g = np.exp(-(x**2) / (2*sigma**2))
        return g / g.sum()
    gauss_coefs = gaussian_kernel(size=7, sigma=1.0)

    # Slice region of interest
    q = train_signal[:, :, 40-win:322+win]  # (B, C, slice_len)

    # Normalize each channel by its mean (per batch & channel)
    q = q / q.mean(axis=2, keepdims=True)

    # Reference spectrum: mean across batch & channels
    q_coef = q.mean(axis=(0,1))  # shape (slice_len,)

    # Copy ROI for smoothing
    t_smooth = train_signal[:, :, 40-win:322+win].copy()

    # Loop over wavelengths inside ROI
    for l in range(win, t_smooth.shape[2]-win):
        coefs = q_coef[l-win:l+win+1] / q_coef[l]              # (2*win+1,)
        window = train_signal[:, :, 40-win+l-win:40-win+l+win+1]  # (B, C, 2*win+1)
        
        # Weighted Gaussian smoothing
        t_smooth[:, :, l] = np.tensordot(window * coefs, gauss_coefs, axes=([2],[0]))

    # Trim edges and reverse order
    if win > 0:
        t_smooth = t_smooth[:, :, win:-win][:, :, ::-1]
    else:
        t_smooth = t_smooth[:, :, ::-1]

    # Concatenate first column (wavelength=0) back
    first_col = train_signal[:, :, 0:1]  # (B, C, 1)
    return np.concatenate([first_col, t_smooth], axis=2)
```
Then we do normalization afterwards.

## **Adjust pipeline from physical principle:**
Dark current represents an **additive noise component** in the measured signal. It should therefore be subtracted **before nonlinearity correction**, which is designed to operate on the *true physical signal*. Including dark current in the nonlinear transformation would violate this assumption and introduce systematic errors.
[formula]

**Benefit:**
Applying dark current subtraction before nonlinearity correction ensures that the polynomial correction operates on a cleaner signal, free from additive noise. This prevents the dark current component from being **amplified or distorted by the nonlinear transformation**, resulting in more accurate signal calibration and improved downstream performance.



## modelling
Method 1 : NN with filter kernels, CNN backbone, and projection layer.

Method 2: transit model and tree model stacking

# Visualization
Shown below are examples of typical anomalous samples frequently observed in the dataset.

## distribution outliers
[distribution_outliers_examples]

## extreme values 
[extreme_values_examples]

## gradient anomalies 
[gradient_anomalies_examples]

## phase anomalies 
[phase_anomalies_examples]

## stability anomalies 
[stability_anomalies_examples]


## transit anomalies 
[transit_anomalies_examples]



## Postprocessing
We didn't figure out robust postprocessing.


# Conclusion

In this work, we built upon the official baseline by introducing a series of targeted improvements across data preprocessing, augmentation, modeling, and calibration. This competition reinforced the importance of careful data preprocessing, the combination of domain knowledge with machine learning, and iterative experimentation. 

## **Cheers to all Kagglers, and really have fun!**
