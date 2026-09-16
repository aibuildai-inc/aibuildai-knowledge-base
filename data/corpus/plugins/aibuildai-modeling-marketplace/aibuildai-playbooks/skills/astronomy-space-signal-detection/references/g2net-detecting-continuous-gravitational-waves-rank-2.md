# 2nd Place Solution: GPU-Accelerated Random Search

Competition: g2net-detecting-continuous-gravitational-waves
Rank: #2
Source: https://www.kaggle.com/c/g2net-detecting-continuous-gravitational-waves/discussion/376504

Thanks to the host and Kaggle staff for holding the competition and congratulations to the winners! I also appreciate my teammates ( @charmq and @yoichi7yamakawa) a lot.

Eight hours before the contest ended, we realized that real data in close frequency range might have exactly the same noise, and by actually identifying some of them, we boosted the performance to 0.855/0.849 (1st in public LB!). In this post, we focus on our main solution without this leak magic, which could still win 2nd place (0.835/0.826).

## Basic Algorithm
After struggling with training neural network models, in the last two weeks we found that a very simple solution could work: a random search of signals. Using the velocity for each timestamp computed by PyFstat, the shapes of waves are determined by four parameters: f0, f1, alpha, and delta. We searched the combination of these parameters that maximizes the mean powers (=square of absolute values) of the corresponding part in each data. The essential part of our algorithm is simple as follows (NumPy-like pseudocode):

```python
stft_sq: (360, n_timestamp)
frequency_Hz: (360)
velocity: (3, n_timestamp)

for _ in range(n_random_search):
    f0, f1, alpha, delta = random_sample_params()
    signal = calc_signal_shape(f0, f1, alpha, delta, velocity)  # (n_timestamp): frequency for each timestamp
    frequency_idx = np.round((signal - frequency_Hz[0]) / (frequency_Hz[1] - frequency_Hz[0]))  # (n_timestamp)
    signal_part = stft_sq[frequency_idx, np.arange(n_timestamp)]  # (n_timestamp): powers of corresponding part in data
    score = np.sqrt(signal_part.mean())
```

As a prediction, we simply outputted the mean score of the two detectors (L1 and H1) for each data.

We implemented batch-wise execution of this algorithm using [Cupy](https://github.com/cupy/cupy) to accelerate on GPU. It enabled the search of 3276800 points in around 20 seconds per data on NVIDIA V100. Therefore, it took around 3 GPU hours and 2 GPU days for the execution of all train data and test data, respectively.

## Details
We tried several frequency widths around signals (we take only the nearest one in the above pseudocode) and weighting methods. The best one was the nearest two and linear interpolation by the differences from signal frequencies.

When calculating signal shapes, we fixed `tref` for the whole data to a certain value because it can be covered by changing f0 and f1.

Choosing the right parameter distribution was important. From the analysis of train data, we realized that data with higher scores than a certain threshold (around 1.541) were almost surely positive. The distribution of found parameters in test data with high scores are as follows  (the right figure is f0 scaled by each data's frequency range):

From this, we decided to sample each parameter from the following distributions, which enhanced the performance a lot:
```
alpha: Uniform([0, 2 \pi])
delta: arcsine(Uniform([-1, 1]))
f0: Beta(2, 2) between frequency range 20% extended to both sides
f1: 1/3 are from -2 * 10^(Uniform([-11, -9])), 2/3 are from 2 * 10^(Uniform([-11, -8]))
```

Around 1/5 of the test data included real noise with nonstationarity and peak in certain frequencies, etc. For these data, we performed time-wise normalization after a simple rule-based frequency mask like below:
```python
def remove_freq_peak(stft_sq, median_coeff: float = 1.1, percent: int = 75):
    freq_std = np.std(stft_sq, axis=1)
    error_freq = (freq_std > np.median(freq_std) * median_coeff) & (freq_std > np.percentile(freq_std, percent))
    stft_sq[error_freq] = stft_sq[~error_freq].mean(axis=0)
    return stft_sq
```
It seems most of the strong noises are removed by this preprocessing (an example of `id:56b090eaf` is below), but we did not have enough time to put much effort into this, so there might be some room for improvement.


## Validation
Since the number of training data was very limited, we calculated the average AUC of 5 different random seeds. It correlated with the public LB to some extent. The best mean train AUC was 0.902940, which seems to overfit a little, still. (Did not have enough time after identifying parameter distribution...)

For real data, we generated a validation set by adding signals to low-score samples, but it did not correlate well with LB, so we focused more on LB scores.

## Things that might be improved
- More sophisticated preprocessing of real data
- 2-stage search
- Rule-based or machine learning postprocessing of the search results
- Consider amplitude differences by timestamps
