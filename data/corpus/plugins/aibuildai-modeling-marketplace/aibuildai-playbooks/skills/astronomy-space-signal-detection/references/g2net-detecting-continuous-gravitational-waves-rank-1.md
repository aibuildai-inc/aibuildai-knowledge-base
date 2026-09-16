# 1st place solution: Summing the power with GPU

Competition: g2net-detecting-continuous-gravitational-waves
Rank: #1
Source: https://www.kaggle.com/c/g2net-detecting-continuous-gravitational-waves/discussion/375910

I thank Kaggle and the organizers for hosting this gravitational wave competition. I enjoyed the previous binary back hole merger competition, as well, and was impressed a lot by the gold-medal solutions. I thought large pretrained image models would be the strongest anyway, but they outperformed with 1-dimensional convolutional neural networks. I joined this competition so that I could build such deep neural network models detecting the wave, not the power, ... but failed.

The largest difference between the two competitions is that the Earth rotates during 120 days and it imprints complicated frequency pattern into the frequency. Adding wave is very delicate, requires very accurate phase patterns, and I was not able to add up the complex Fourier modes effectively within reasonable computational resources. After more than one month without any progress, I thought I should get a silver medal even with an unsatisfactory approach, give up adding the wave and add the power.

# Solution

- Sum power (absolute-value squared) along various signal patterns
- No machine learning
- No use of external data or leakage

## Power summation

1. Extract signal frequency and amplitude [total power P(t)] from the simulations
2. Subtract Doppler shift frequency from the data frequency for 4000 signal patterns 
3. Weight the data proportional to the signal amplitude pattern; this is the optimal linear weight w(t)
4. Sum the weighted power along lines: 360 frequencies (intercept) × 241 slops in [-120, 120] (frequency bin / 120 days)
5. Take the maximum



The values are highly skewed from the typical range [-1, 1] because these are maxima of 4000 templates × 360 frequencies × 241 slops.

This took 5 days using GPU RTX 3090. Full range of slope is [-360, 360] but [-120, 120] already took long enough.

##  Real noise normalization

1. Normalize by the noise rms at each time h -> h / sigma(t)
2. Remove single-frequency noise by masking anomalously large frequency bin
3. Normalize the frequency dependence by the remaining rms sigma(f)

I know the noise is not always written as a product of time dependence and frequency dependence, but I did not have time for better treatment. Some false positives are remaining in my prediction.

## Follow up summation with sinc kernel

The signal spread among frequency bins with sinc function (assuming the window function for the short-time Fourier transform is almost top hat). I use a sinc kernel with width 8 and stride 1/8 frequency bin to collect the signal. This is the optimal linear weighting in the frequency direction. I recompute the power sum with this kernel around the largest-power line in the first step for a subsample of 400 templates. This gives a surprisingly large boost to the public score 0.825 -> 0.848

Finally, I apply a sigmoid to the standardized power sum and submit, which is same as just submitting the power sum. I thought the prediction value must also depend on the noise level; if the signal is undetected, there should be larger possibility to be positive for larger noise because more data are undetected. I modeled this effect but could not improve the score. 

PS:
More about the weight
https://www.kaggle.com/code/junkoda/optimal-weights-for-signal-to-noise-ratio

Code
https://github.com/junkoda/kaggle_g2net2_solution
