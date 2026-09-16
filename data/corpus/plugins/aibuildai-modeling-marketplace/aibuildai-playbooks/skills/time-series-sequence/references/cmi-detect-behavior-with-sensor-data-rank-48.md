# 48th Place Solution

Competition: cmi-detect-behavior-with-sensor-data
Rank: #48
Source: https://www.kaggle.com/c/cmi-detect-behavior-with-sensor-data/writeups/48th-place-solution

First of all, thank you for organizing such a fantastic competition! Kudos to the organizers and all the participants for their hard work!

Here’s a brief overview of my approach.

## Models Used
I used an ensemble of transformer, GRU, and squeezeformer models. The models were divided into two categories: IMU only and Full data. For Full data, I used only GRU models. Below is the structure of the GRU model (other models are quite similar).

[GRU_model]

A key feature was the addition of a 1D CNN layer after the input, rather than using GRU alone. This significantly boosted the cross-validation (CV) score.

### Best Submission (Overview)
IMU only: 2 transformer models + 2 GRU models + 1 squeezeformer model

Full data: 10 GRU models

## Feature Engineering
Since this is time-series data, I mainly added features that incorporate information beyond the current row. For example, applying a rolling standard deviation to acc_x improved CV. I also included centered rolling statistics:

```python
train_df['acc_x_std'] = train_df.groupby('sequence_id')['acc_x'].transform(lambda x: x.rolling(window=5, min_periods=1).std())
train_df['acc_x_center_std'] = train_df.groupby('sequence_id')['acc_x'].transform(lambda x: x.rolling(window=5, center=True, min_periods=1).std())
```

Interestingly, something unexpected happened: I accidentally added a feature from thm twice, and for some reason, CV improved by 0.01. Although the reason is unclear, I kept it since it enhanced performance.

## EMA
Incorporating EMA helped improve CV while also enhancing generalization.

## Null Handling
The tof data contained a fair amount of missing values, so proper null handling was essential. I used linear interpolation and median imputation to fill in missing values smoothly. Here’s the code used for null handling during training:

```python
for group_prefix in tof_groups:
    group_cols = [col for col in train_df.columns if col.startswith(group_prefix)]
    train_df[group_cols] = train_df[group_cols].replace(-1, np.nan)
    train_df[group_cols] = train_df[group_cols].apply(
        lambda row: row.interpolate(method='linear', limit_direction='both', limit=5), axis=1
    )
    train_df[group_cols] = train_df[group_cols].apply(
        lambda row: row.fillna(row.median()), axis=1
    )
    train_df[group_cols] = train_df[group_cols].fillna(-1)
```

## Mixup
I set the mixup alpha to 1.0. This was intended to improve generalization, and it did lead to better CV.

## Best Models and Final Models
Models with early stopping were treated as “best models,” while those without were considered “final models.” Both types were used in the ensemble. Here’s the breakdown:

IMU only:

2 transformer models (best & final)

2 GRU models (best & final)

1 squeezeformer model (final only)

Full data:

10 GRU models (2 versions × 5 seeds: best & final)

## Ensemble Method
I used simple averaging for the ensemble.

## What Didn’t Work
Auxiliary Loss for orientation: It slightly improved CV but worsened the leaderboard (LB) score, so I didn’t adopt it.



Thanks for reading!
