# 34th place solution

Competition: ariel-data-challenge-2024
Rank: #32
Source: https://www.kaggle.com/c/ariel-data-challenge-2024/discussion/543676

# Preparing the Transit Signal

(Based on [this notebook](https://www.kaggle.com/code/pourchot/ariel-data-challenge-2024))

- Applied a lowpass filter instead of binning
- Downsampled to 232 points
- Computed a moving average across wavelengths with a window size of 31
- Extracted and flipped the transit signal before optimization
[ref. 1](https://www.kaggle.com/competitions/ariel-data-challenge-2024/discussion/538139#3027343), [ref. 2](https://www.kaggle.com/competitions/ariel-data-challenge-2024/discussion/529412#2965289)
- Use joblib Parallel, delayed

Applied a lowpass filter instead of binning, Downsampled to 232 points

```python
import scipy.signal as sci_signal

def butter_lowpass(cutoff, fs, order=5):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = sci_signal.butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def apply_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = sci_signal.filtfilt(b, a, data)
    return y

def downsampling(signal, interval):

    cut = int(500 / interval)
    signal = signal[::interval, :]
    signal = signal[cut:-cut, :]
    return signal
```

```python
    for i in range(binned.shape[1]):
        binned[:, i] = apply_lowpass_filter(binned[:, i],  0.005, 1)

    # downsampling
    binned = downsampling(binned, config.interval)
```

Computed a moving average across wavelengths with a window size of 31

```python
def moving_average(arr, window_size):
    return np.convolve(arr, np.ones(window_size)/window_size, mode='same')

window_size = 31

for i in tqdm(range(pre_train.shape[0])):
    pre_train[i, :, 1:] = np.apply_along_axis(moving_average, axis=1, arr=pre_train[i, :, 1:], window_size=window_size)
```

Extracted and flipped the transit signal before optimization

```python
pre_train = np.concatenate([pre_train[:, :, [0]], np.flip(pre_train[:, :, 39:321], axis=2)], axis=2)
```

# Optimization

The optimization process estimates (R/R)^2, the baseline (representing planet radius), and the concentration of each gas molecule as follows:

- Set initial values for the baseline and concentrations of each molecule
- Construct the complete signal by summing the baseline and the individual signals for each gas molecule (each molecule’s signal was collected from ExoMol data)
- Calculate RMSE by comparing the depth of the generated signal to the observed transit signal across all wavelengths
- Use joblib Parallel, delayed

```python
def make_signal(s_list):
    """make signal
    s_list: baseline and concentrations of each molecule
    """
    result = np.repeat(np.array([s_list[0]], dtype=defalut_dtype), spectrum_array.shape[1])
    for i, p in enumerate(s_list[1:]):
        result += (spectrum_array[i, :]**2) * p
    return result
    
    
def objective_each_signal(s, signal, p1, p2):

    best_q = 1e10

    for i in range(4) :
        delta = int(150 / config.interval)

        x = np.arange(signal.shape[0], dtype=defalut_dtype)
        y = signal.copy()

        y = np.concatenate([y[:p1-delta], y[p1+delta:p2-delta]* (1 + s), y[p2+delta:]])
        x = np.concatenate([x[:p1-delta], x[p1+delta:p2-delta], x[p2+delta:]])

        z = np.polyfit(x, y, deg=i)
        p = np.poly1d(z)
        q = np.mean((p(x) - y)**2)
        
        if q < best_q :
            best_q = q
        
    return best_q

def objective(s_list, signals, p1, p2):

    spectrum = make_signal(s_list)
    mae = np.mean([objective_each_signal(spectrum[wl], signals[:, wl], p1, p2) for wl in range(1, signals.shape[1], 2)])
    return mae

def optimize_signal(i, phase_dict, pre_train):
    
    p1, p2 = phase_dict[i]

	  # baselline and amount of each gas molecules
    # ('1H2-16O','12C-1H4','12C-16O2','12C-16O','14N-1H3','1H-12C-14N','1H2-32S','48Ti-16O','51V-16O')
    initial_guess = [2.42760318e-03, 1.23843539e-05, 1.62354048e-04, 8.55036534e-05,
                    5.03158911e-06, 2.44615243e-05, 7.56269106e-06, 3.30325790e-05,
                    2.41132168e-05, 2.12473388e-05]
    
    result = minimize(
        objective, 
        initial_guess, 
        args=(pre_train[i, :, :], p1, p2), 
        method='L-BFGS-B', 
        bounds=[(0, None)] * (len(initial_guess))
    )

    spectrum = make_signal(result.x)

    return spectrum
```

# Post-Processing

For `wl_1`, compute a weighted mean of the predicted `wl_1` value and the mean of the other wavelengths.

```python
pred_mean[:, 0] = np.ones_like(pred_mean[:, 1]) * (pred_mean[:, 0] * 0.2 + pred_mean[:, 1:].mean(axis=1) * 0.8)
```

# Determining Sigma

- Calculate sigma based on the RMSE for each transit signal and the mean transit signal across each wavelength.
- Clipping to 1 * sigma

```python
def calc_rmse_per_wl(gt, pred):
    rmse_list = []
    for wl in range(pred.shape[1]):
        rmse_list.append(mean_squared_error(gt[:, wl], pred[:, wl], squared=False))
    return rmse_list

sigma_values = np.array(calc_rmse_per_wl(train_labels.values, pred_mean))
const = 1.4
sigma_const = 1
sigma_values = sigma_values * const

sigma = np.std(sigma_values)
mean = np.mean(sigma_values)

sigma_values_clipped = sigma_values.copy()
sigma_values_clipped = sigma_values_clipped.clip(-sigma_const*sigma+mean,sigma_const*sigma+mean)

pred_sigma_wl = np.repeat(sigma_values_clipped.reshape(1, -1), len(adc_info), axis=0)
```
