# 9th place solution

Competition: LANL-Earthquake-Prediction
Rank: #9
Source: https://www.kaggle.com/c/LANL-Earthquake-Prediction/discussion/94609#latest-545993

After discovery of possible (at that point) p4677 related leak, I decided to drop from competition. Heading into last week of competition I had clear idea how I would exploit it. This post [Will this decide who wins?](https://www.kaggle.com/c/LANL-Earthquake-Prediction/discussion/94086) and the fact I had nothing to lose (dropped to ~1K public place) if assumption about test set distribution was wrong I decided to implement the idea.

I wanted to use Leave K-EQ out type of CV scheme, but thanks to @cpmpml we know that difference between mean of validation and train could lead to under/overfit, so if mean of all EQ is the same, I think, it might work.

I transformed target to normalized time to failure multiplied by twice the estimated mean of private set. I had 2 values (2 submissions): 6.3 and 5.9. In hindsight 5.9 was way off, but my estimation process was quiet crude. Late submission with higher value have slightly better score.

[target]

I chose LGBM, as I had been using it prior to leak discovery. 
Features were peaked from public kernel ["Even more features"](https://www.kaggle.com/artgor/even-more-features) by @artgor, according to past models feature importance and ["My Top 30 Features"](https://www.kaggle.com/c/LANL-Earthquake-Prediction/discussion/93148) by @scirpus
```
num_peaks_10,
num_crossing_0,
percentile_roll_std_5_window_100,
abs_percentile_80, 
fftr_percentile_roll_std_80_window_10000
```
Folds were generated using this code:
`for val_t in itertools.combinations(range(15), 2):`
where `val_t` is two EQ indexes for validation. Only full EQ were used.
All these were enough for 9th place.

More interesting for me were ideas that didn't work.
One of them was effort to predict not ttf, but a pair of period of EQ and normalized time to failure. I did try siamese network based on CNN1D with similarity metric:
`np.exp(-np.abs(right_period - left_period)/c1) * np.exp(-np.abs(right_norm_ttf - left_norm_ttf)/c2)`
constants `c1` and `c2` were to limit points that are close enough (hyperparameters). I did not fully explore this solution, but it was fun to work on.
I did try other approaches but I strongly believe that period of EQ is not predictable.
