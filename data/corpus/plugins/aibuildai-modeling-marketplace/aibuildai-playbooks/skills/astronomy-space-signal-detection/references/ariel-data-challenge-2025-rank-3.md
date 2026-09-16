# 3rd Place Solution

Competition: ariel-data-challenge-2025
Rank: #3
Source: https://www.kaggle.com/c/ariel-data-challenge-2025/writeups/3rd-place-solution

# Introduction & Summary

Huge thank you to the organizers as well as the staff for this amazing competition! I learnt a lot about data engineering and modelling techniques and I look forward to competing in the future!

My approach contain those steps:

- Step 1: Preprocessing & Calibration
- Step 2: Signal extraction
- Step 3: Feed into CNN for additional feature extraction
- Step 4: With signal features and CNN features, feed into ensemble of rational quadratic neural network

# Preprocessing & Calibration

I used the standard calibration techniques as given in the competition itself. It includes the calibration for gain/offset correction, and manages non-linearity, dark, dead and hot pixels. Additionally, at the end of the calibration step, I remove signals that are more than $8 \sigma$ away in the time dimension. This addition alone adds ~0.05 to my CV (which is just a simple 80-20 split, turning out to be almost same as LB; LB feedback time is 5-8 hours). I couldn't spend much time on this section as rerunning this for all data, even with joblib, takes 1+ hour.

```python
mean = np.nanmean(signal, 0)[None, :, :]
std = np.nanstd(signal, 0)[None, :, :]
signal[(signal > mean + 8.0 * std) | (signal < mean - 8.0 * std)] = np.nan
```

The above is the exact added code used in inference.

Given features like `Rs` were included as well as minor feature engineering, like `Rs ** 2`, but they don't seem too important.

# Signal Extraction

Phase extraction is done using extrema of gradient.

1. The calibrated signal is divided by the mean of itself
1. A simple moving average of period k is applied to the signal (k = 88 for airs and 655 for fgs1, found using optuna)
2. A set of moving averages is then applied to the gradient of the signal, in which p,q = argmin,argmax are extracted
3. p,q are then moved to the left,right until their gradients are within some interquartile range of all gradients so they cover all of the transit phase

The features are then extracted via simple mean/min/max/range/min div edge (approx transit depth)/polyfit to 8th degree etc. of each of the phases. Additionally, the airs data is `array_split`'d along the frequency dimension (size 356) into 32 chunks. Each chunk then gets processed like above in the same way, and their features added to the list of features we have. This adds quite a bit of resolution and adds ~0.025 CV.

# Feed into CNN

The CNN doesn't care about the phases; the input is simply the airs and fgs1 signal after the first moving average, split into 92 chunks across time with each chunk averaged. Additional steps might be adding phase signal into the CNN, as well as separating airs/fgs1 data into different subnets since they might not be time aligned, but ideas like these were not implemented

There are many instances of CNN, as they are integrated into the RQ NN ensemble. To prevent overfitting, almost all layers of the CNN have the same weight and shape. No activation function other than max pool was used.

# Ensemble of Rational Quadratic NN

## Preparing the features

One idea was borrowed from https://arxiv.org/pdf/2407.04491.

- I used same feature scaling but with $\frac{x}{\sqrt{18+x^2/9}}$

## Training

The idea is deep ensembling where each instance in the ensemble is trained the exact same way, just with different weight initialization. We just straight up predict everything, the `pred` and `sigma` with our model, no sampling or quantile regression.

- Optimizer is adabelief with (b1=0.999, b2=0.9998)
- Cosine one cycle learning rate schedule with square root shaping
- 36 ensembles, with 88 RQ clusters

Intuitively, I chose soft RBF/RQ (they perform similarly, both adds ~0.15 CV) because there were many transit curve shapes (it was mentioned that many different models of limb darkening were added). So this soft RQ aims to first sort instances into soft groups, and then perform simple linear regression within. The image shows the distribution of the first RQ cluster of the first instance.

The training is quite unique; instead of training on average of losses (competition metric), I spent more effort training models which are performing worse. Similar to evolution, the best models don't need to get better anymore. This avoids overfitting and improves my CV by ~0.03+.

```python
m = jnp.median(losses, 0) # 0 is ensemble dimension
w = jax.nn.relu(losses - m)
w = w / w.sum(0, keepdims=True) # scale so every channel is treated equally
w = jax.lax.stop_gradient(w)
return (losses * w).sum() / w.sum()
```

The above is the exact code used in training. Only values worse than the median will get trained.

# What didn't work & Conclusion

- L1, L2 regularisation.
- Dropout (works well in very specific circumstances; unstable).
- Data augmentation via random splicing of transit data (although I saw there were some instances where the phase is cut off).
- Fischer-information-like or entropy based regularisation for RBF/RQ, to encourage more uncertainty of groups
- Full trainable Mahalanobis distance

I used both observations for the same planet if they exist during training, but only the first observation during inference. The NN inference is instant and the data reading takes up the majority of the time.

In conclusion an acceptable result was achieved with little or no understanding of the underlying physics nor advanced drift/transit window denoising techniques.
