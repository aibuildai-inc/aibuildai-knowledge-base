# 7th Place Solution - Part 1: Removing Noise from Signal

Competition: liverpool-ion-switching
Rank: #7
Source: https://www.kaggle.com/c/liverpool-ion-switching/discussion/154011

Congratulations to all winners!

First of all i want to thank my wonderful teammates @philippsinger, @dott1718 and @cdeotte for the awesome collaborative teamwork. Everyone gave so much effort to get the best result and I am already looking forward to teaming-up with you again.

In this write-up I will get into details of our routine to remove noise from the provided signal. Afterwards, @cdeotte will go into more detail about the models that we used in our final submission. See his discussion [here](https://www.kaggle.com/c/liverpool-ion-switching/discussion/154253).

# Data Cleaning

One of the most difficult but at the same time also one of the most rewarding things to do in this challenge was the cleaning of the dataset (train and test). Prior to teaming-up, I had exclusively worked on that part and only used public kernels such as the WaveNet kernel. Thank you @siavrez for this kernel! I also found HMM (just straightforward hmmlearn) to be a good choice for quickly testing the cleaning but we didn’t use HMM in our final solution.

The original data could be divided into batches of 10 s or 50 s and the “signal” already showed a pearson correlation of 0.8017 with “open_channels”.





For ease of understanding, I will refer to parts of the train and test set with their corresponding maximum open channels and opening probability or their type by "color". The “pink” parts in the test set are assumed to all have a maximum of 5 channels with low opening probability.

## Sine Drift
Early on in the competition, an artificial sine drift added to parts of the train and test set has been spotted (https://www.kaggle.com/c/liverpool-ion-switching/discussion/133874).
The drift was identified to be a 50 s long half sine wave (or just the first 10 s of that) with an amplitude of 5 and may be removed with the following code snippet:

```
def sine_func(x, a, b):
    return -a * np.sin(b * x)

sine_drift = sine_func(np.linspace(0, 499999, 500000), -5, np.pi / 500000)
sine_drift_short = sine_drift[:100000]
# Clean the train set
train_csv.loc[train_csv["100k_batch"] == 5, 'signal'] -= sine_drift_short  # batch 2 first segment
train_csv.loc[train_csv["500k_batch"] == 6, 'signal'] -= sine_drift
train_csv.loc[train_csv["500k_batch"] == 7, 'signal'] -= sine_drift
train_csv.loc[train_csv["500k_batch"] == 8, 'signal'] -= sine_drift
train_csv.loc[train_csv["500k_batch"] == 9, 'signal'] -= sine_drift
# Clean the test set
test_csv.loc[test_csv["100k_batch"] == 0, 'signal'] -= sine_drift_short
test_csv.loc[test_csv["100k_batch"] == 1, 'signal'] -= sine_drift_short
test_csv.loc[test_csv["100k_batch"] == 4, 'signal'] -= sine_drift_short
test_csv.loc[test_csv["100k_batch"] == 6, 'signal'] -= sine_drift_short
test_csv.loc[test_csv["100k_batch"] == 7, 'signal'] -= sine_drift_short
test_csv.loc[test_csv["100k_batch"] == 8, 'signal'] -= sine_drift_short
test_csv.loc[test_csv["500k_batch"] == 2, 'signal'] -= sine_drift
```




## 10 Channel Offset (“Ghost-Drift”)
The next thing that becomes obvious in the data, was a large offset between the 10 channel batches and the rest of the data. We now know where this difference originates from (leak), but at the time of evaluating, we were not sure about the exact offset and we have used various techniques for calculating it. Depending on the technique, this offset was calculated to be between -2.69 and -2.76. By removing that offset, neural nets may have an easier time at prediction by cross learning from other batches. 

A straightforward method for removing the offset is using a linear regression for the 10 channel batches (open_channels vs signal) and for the rest of the batches. 

```
def linear_fx(x, m, y0):
    return (m * x) + y0

xdata = train_csv_batch10["open_channels"]
ydata = train_csv_batch10["signal"]
popt1, pcov = curve_fit(linear_fx, xdata, ydata, method='lm')

xdata = train_csv_batch5["open_channels"]
ydata = train_csv_batch5["signal"]
popt2, pcov = curve_fit(linear_fx, xdata, ydata, method='lm')

static_diff = popt1[1] - popt2[1]
print("Offset 10c to rest:", static_diff)
```

While this method neglects possible differences in slope and minor offsets between batches, it gave a robust first guess of the actual offset.

Later on, @dott1718 developed a Gaussian Mixed Model for fitting the Gaussians of each batch using equal spacings and a binomial distribution. In our cross-validation we could not clearly identify which was doing better. 

When adding a second dimension to this GMM one can clearly identify a correlation between previous states and following states. It appears, that the signal has a decay that is within the magnitude of the recording frequency of 10 kHz. That means, if a previous signal was larger than the current signal, the current signal will be slightly higher than expected. We assume this to be an artifact from the signal recording or from the electrical ion channel model (electrical circuit) and tried to make use of that feature (other called it memory) in any way and improve out models on that, but to our amazement, WaveNets (and probably many other ML models) seem to learn that behavior already on their own and any additional cleaning is not yielding improvements. We did gain a slight improvement in our test HMM using that knowledge. 




## Outlier removal
In the first 500k batch and in the 8th 500k batch of the train set, severe outliers have been spotted. 

The outlier in the first batch, we simply replaced by gaussian noise with an appropriate mean and variance. 

```
FIRST_OUTLIER = (47.858, 47.8629)
noise = np.random.normal(loc=-2.732151, scale=0.19, size=50)
train_csv.loc[(train_csv.time &gt;= FIRST_OUTLIER[0]) &amp; (train_csv.time &lt;= FIRST_OUTLIER[1]), "signal"] = noise
```

The second sequence with outliers, we just removed for training.
```
SECOND_OUTLIER = (364.290, 382.2899)
train_csv = train_csv.loc[(train_csv.time &lt; SECOND_OUTLIER[0]) | (train_csv.time &gt; SECOND_OUTLIER[1]), :]
```


## Zero Level Offset

Just like the 10 channel offset, we have a zero level offset when no channels are open. Simple linear regression can be used to shift the datasets from ~2.73 and align the “zero channel” with zero mean. We can also use the results of linear regression to extract the scaling between “signal” and “open_channels” which allows us to extract the noise from the train signal and study it. 


## 50 Hz Line Noise

When plotting the FFT of the signal and of the noise, we can clearly see that most of the noise seems to be white gaussian noise. But we can also identify some sharp peaks at 50 Hz, ~1150 Hz, ~1250 Hz, ~1450 Hz and ~3130 Hz. In Liverpool, they use 50 Hz in their power Lines and applying a narrow hann window to attenuate that frequency greatly improved our modeling capabilities. For best results, we tuned the attenuation with regard to signal to noise ratio at 50 Hz (higher attenuation for slow open channels compared to the fast open channels). With a grid search, we optimized the parameters "hann windows width", "attenuation high" and "attenuation low".
We also tried removing of the other peaks, that are somewhat random in the data, but without ever getting a significant improvement in our CV score. 










## Minor Offsets (mostly in test set)

When leveling all batches with the same static offset mentioned above (zero level offset), we still experience significant differences between the batches. The very first 100k of the training set seem to be shifted about 0.03 in positive direction.




In the test set, those shifts are much more frequent and careful removal of those minor offsets appeared to have significant influence on the predictions of our models. To remove the offset of the training set we can easily use the ground truth values of open_channels combined with the scaling factor from above. As we don’t have that information in the test set, we evaluated three different techniques to remove that offset and they all led to similar results.
To begin with, we have used old predictions for the test set, that scored high on the leaderboard to calculate the noise in the test set and ultimately the minor offsets. As this is a delicate thing, especially in the 10 channel batches with the high variance, we also checked how those values are changing by recursively using the predictions for calculation of the noise. As expected, the difference was somewhat significant in the 10 channel batches and almost non-existent in the remaining batches. To archive a more robust solution, we later based the entire shifting routine on the GMM mentioned above, modelling the gaussians with equal spacings and binomial distributions. 






## Extra Data

We used 700k rows of annotated data (3 channels and 5 channels) from Richards GitHub (https://github.com/RichardBJ/Deep-Channel/tree/master/dataset) which yielded a tiny improvement on CV. All the above steps have also been applied to that extra data. 


## What didn't work

- Creation of additional synthetic training data
- Unequal spacing between gaussians
- Kalman Filtering
- Removal of ~1150 Hz, ~1250 Hz, ~1450 Hz and ~3130 Hz noise
- Prior modelling of the signal decay rate before feeding it to WaveNet
- Unequal scaling of batches
- Single models for each “type” of batch
