# 22th-place solution : simulated CW signals & augmentations

Competition: g2net-detecting-continuous-gravitational-waves
Rank: #22
Source: https://www.kaggle.com/c/g2net-detecting-continuous-gravitational-waves/discussion/375927

Hello everyone!

First, thanks to EGO for hosting an exciting competition! Also, congratulations to all the winners!

## Data

### Pre-Processing

In my experiment, [preprocessing](https://www.kaggle.com/code/laeyoung/g2net-large-kernel-inference) (`normalize` function) works better than the power spectrogram. It improves the score by about +0.02 on CV/LB. After normalizing the signal, take a mean over the time axis. The final shape is (360, 360).

### Simulation

Generating samples is the most crucial part of boosting the score. I can get 0.761 on the LB with a single model.

In short, signal depth (`sqrtSX / h0`) takes a huge impact. I generated 100K samples (50K positives, 50K negatives) and uniformly sampled the signal depth between 10 and 100. `cosi` parameter is uniformly sampled (-1, 1).

| signal depth  | LB score  |
| --- | --- |
| 10 ~ 50 | 0.73x ~ 0.74x |
| 10 ~ 80 | 0.75x |
| 10 ~ 100 | 0.761 |

### Augmentations

Also, I've worked on the augmentations for much time. Here's a list.

1. v/hflip
2. shuffle channel
3. shift on freq-axis 
4. denoise a signal (subtract corresponding noise from the signal)
5. add noises
  * Guassian N(0, 1e-2)
  * mixed (add or concatenate) with another (stationary) noise(s)
6. add vertical line artifact(s).
7. SpecAugment
8. mixup (alpha 5.0)
  * perform `or` mixup

## Model

First, I tried to search for the backbones (effnet, nfnet, resnest, convnext, vit-based) and found `convnext` works best on CV & LB score. After selecting a baseline backbone, I experimented with customizing a stem layer (e.g. large kernel & pool sizes, multiple convolutions stem with various kernel sizes) to detect the long-lasting signal effectively, but they didn't affect the performance positively.

## Ensemble

Most of the models used at the ensemble are `convnext-xlarge` but each model trained with different variances (e.g. augmentations, simulated samples, ...) and `eca-nfnet-l2`, `efficientnetv2-xl` for one model. Every model trained on various datasets and LB score seems reliable, so I adjusted the ensemble weights by LB score.

I selected the two best LB submissions (LB 0.768 PB 0.771). And the best PB that I didn't select is 0.778 (LB 0.766) (mixing all my experiments).

## Works

* `convnext` family backbone
* signal depth 10 ~ 100
* hard augmentation
* pair stratified k fold
  * 8 folds
  * stratified on the target
  * `pair` means the pair (corresponding noise & signal) must be in the same fold.
* pseudo label (smooth label)
* segmentation (but hard to converge on my experiment)
* TTA

## Not Works

* segmentation with classification head (0.6 * bce + 0.4 * dice)
  * Actually, seg with cls works slightly better than only cls, but hard to train without loss divergence. So, I just did only cls.
* `cosi == 0`
  * `cosi` is also a critical parameter to determine an SNR. I generated more samples where `cosi` is 0, but there's a score drop.
* augmentations (not worked)
  * swap with random negatives (proposed at the past competition)
  * random resized crop
* Customize a stem layer with large kernel & pool sizes.

I hope this could help you :)

Happy new year!
