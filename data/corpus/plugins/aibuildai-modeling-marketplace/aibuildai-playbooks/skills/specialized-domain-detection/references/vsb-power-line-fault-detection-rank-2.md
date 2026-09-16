# 2nd Place Solution

Competition: vsb-power-line-fault-detection
Rank: #2
Source: https://www.kaggle.com/c/vsb-power-line-fault-detection/discussion/86616#latest-501584

Congratulations to all the winners! 
And thanks a lot to VSB/Enet Center and Kaggle for this exciting competition.

## Here are the details about my single model:
Preprocessing:
Before extracting features, I used DWT for noise reduction, and I think this is helpful for the stability of the model.[you can find here](https://www.kaggle.com/jackvial/dwt-signal-denoising)

### Features:
I tried a lot of different methods, such as RNN taking 80, 100, 160, 250 time steps, basic features, energy features, peak features, and also tried to automatically extract features from CNN models, different RNN models. Unfortunately, none of these attempts have brought me a big improvement. Finally, the addition of the peak feature on the basic RNN architecture can effectively alleviate the over-fitting. This is the problem that I think is the most important problem to solve. Extracting the peak features using python's signal library, all this features can be found in [here](https://ieeexplore.ieee.org/abstract/document/7909221). Here is the code:

```python
# Extract peak features
from scipy.signal import find_peaks, peak_widths, peak_prominences

def remove_false_peak(signal, p1, p2, maxDistance=10):
    peak_diff = np.diff(p2)
    if len(peak_diff) == 0:
        return p1
    ticks = []
    for i, d in enumerate(peak_diff):
        ratio = signal[p2[i+1]]/signal[p2[i]]
        if d &lt; maxDistance and -0.25 &gt; ratio and ratio &gt; -4:
            ticks.append((p2[i], p2[i+1]))
    mask = np.array([True]*len(p1))
    for i, j in ticks:
        mask = mask &amp; ((p1 &lt; i) | (p1 &gt; 500+j))
    return p1[mask]


def get_peaks(signal):
    p1_1, _ = find_peaks(signal, height=[5, 100])
    p1_2, _ = find_peaks(-signal, height=[5, 100])
    p1 = np.union1d(p1_1, p1_2)
    n_peaks, _ = find_peaks(-signal, height=[10, 100])
    p_peaks, _ = find_peaks(signal, height=[10, 100])
    p2 = np.union1d(n_peaks, p_peaks)
    p = remove_false_peak(signal, p1, p2, maxDistance=10)
    return np.intersect1d(p1_1, p), np.intersect1d(p1_2, p)


def extract_peak_feature(signal):
    p_peaks, n_peaks = get_peaks(signal)

    num_p, num_n = len(p_peaks), len(n_peaks)

    sig_peak_width = np.concatenate(
        [peak_widths(signal, p_peaks)[0], peak_widths(-signal, n_peaks)[0]])
    sig_peak_height = abs(signal[np.concatenate([p_peaks, n_peaks])])

    if num_n or num_p:
        height_mean = sig_peak_height.mean()
        height_max = sig_peak_height.max()
        height_min = sig_peak_height.min()
        height_median = np.median(sig_peak_height)

        width_mean = sig_peak_width.mean()
        width_max = sig_peak_width.max()
        width_min = sig_peak_width.min()
        width_median = np.median(sig_peak_width)

        return np.array([height_mean, height_max, height_min, height_median,
                         width_mean, width_max, width_min, width_median, num_p, num_n])
    else:
        return np.zeros(10)
```

At the same time, the added peak feature has a large number of outliers, which I convert to a missing value. Then perform the missing value processing(dividing the data into groups based on the attribute with the largest correlation coefficient of the missing value, and then calculating the average value of each group. Just put these averages in the missing values.)

### Training:
epochs: 25
Checkpoint monitor='val_loss'

## My final solution was a ensemble of three models:
My single Model
[VSB Competition : Stacked Attention Capsule BiLSTM](https://www.kaggle.com/tarunpaparaju/vsb-competition-attention-bilstm-with-features?scriptVersionId=10690570)
[Handmade features](https://www.kaggleusercontent.com/kf/10818864/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..rLDkCCqNGYG5hfrj7oMt9A.ucfuA1j7MlivrTFzzAvVq7SpDSojFzTdXNVHqC5T7q0Vc4AjG5OP-2Pi0EngziSLz6FHxHoY4lqxvYj02gOtfMh9dGMJnCitLHWZ4JrZX10kzWvvLYhAmbtfm6Mk2ej46868zJzHFQ9RKnvcUjjBNQ.abKNk9CPW8feEC28o41osg/__results__.html)

If you think there is something incorrect or that could be improved, please leave your comments! And thank you everybody for the great kernels and discussions.(From 6th but it is exactly what I want to say)
