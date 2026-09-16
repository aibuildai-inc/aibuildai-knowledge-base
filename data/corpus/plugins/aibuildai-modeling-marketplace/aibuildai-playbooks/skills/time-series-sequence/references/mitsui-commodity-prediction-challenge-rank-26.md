# 26th Place Solution – MITSUI&CO. Commodity Prediction Challenge Writeup

Competition: mitsui-commodity-prediction-challenge
Rank: #26
Source: https://www.kaggle.com/c/mitsui-commodity-prediction-challenge/writeups/26th-place-mitsui-and-co-commodity-prediction

# Introduction
I am very excited to have achieved my first-ever Silver Medal in this competition. I believe the key to this result was identifying features that maintain a stable correlation with the target regardless of the date_id, although I must admit that luck also played a significant role.

# Solution Overview
I transformed the dataset into a "long format" (as shown below) to allow a single model to handle all targets simultaneously.

| date_id | feature_1 | feature_2 | ... | feature_n | target | target_id |
| ---- | ---- | ---- | ---- | ---- | ---- | ---- |
| 0 | 0.34 | 0.57 | ... | 0.23 | 0.15 | target_0 |
| 0 | 0.12 | 0.22 | ... | 0.43 | 0.07 | target_1 |
| ... | ... | ... | ... | ... | ... | ... |

## Model and Features
- Model: LightGBM
- Features: I prioritized features that showed a consistent correlation with the target over time.
   - Rank of the mean of past returns
   - Rank of the standard deviation of past returns
   - Categorical values generated via clustering
   - Flags indicating if the mode of past return ranks was within the top or bottom 12
   - Manually assigned category labels

# Model and Feature Evaluation
## Cross-Validation (CV)
I used a simple hold-out method. I allocated 70% of the dataset (575 days) to the validation set—a period considerably longer than the actual public/private evaluation windows—to ensure robustness.

## Evaluation Metric
I evaluated model performance and feature importance by plotting the cumulative correlation coefficients for each date_id.

- An upward-sloping line indicates a positive correlation.
- A downward-sloping line indicates a negative correlation.
- A line closer to a straight diagonal indicates that the correlation is stable across different date_ids.



# Feature Engineering
I focused heavily on the stability of correlations. Here are three notable features:

## Rank of Mean Past Returns
For a specific target_n, this feature represents its rank (from 0 to 423) based on its mean value across the training data. 

```Python
train_labels = pd.read_csv("train_labels.csv").set_index("date_id").fillna(0)
train_labels = train_labels.rank(axis=1)
train_length = int(len(train_labels.index) * 0.7)
mean = train_labels.head(train_length).mean()
```

As shown in the charts, this feature demonstrates a consistently positive correlation with the target.

- Mean Correlation: 0.035778
- Std Dev of Correlation: 0.192780



## Rank of Standard Deviation of Past Returns
This represents the rank of the standard deviation of target_n across the training data.

```Python
train_labels = pd.read_csv("train_labels.csv").set_index("date_id").fillna(0)
train_labels = train_labels.rank(axis=1)
train_length = int(len(train_labels.index) * 0.7)
std = train_labels.head(train_length).std()
```

While this feature often shows a negative correlation, it is considerably more unstable compared to the "Mean Rank" feature.



## Clustering Categories
By treating each target's time series as a vector and applying dimension reduction via UMAP, the targets clearly separated into two distinct clusters. Interestingly, these clusters often exhibited an inverse relationship (when one cluster went up, the other went down).

I used the cluster IDs from K-Means as a categorical feature, which slightly improved the model's performance. I suspect there might have been a more sophisticated way to utilize this discovery, but I could not.



# Model Performance
The prediction accuracy (competition's metrics) of my final model is as follows:

- Train (Blue): 0.1787
- Valid (Orange): 0.2047



The validation score remained stable and even outperformed the training score in terms of cumulative trend, suggesting good generalization.

# Conclusion
My approach centered on a LightGBM model built with time-stable features. While I discovered a clear bifurcation in target behavior through clustering, I couldn't fully exploit that insight beyond a simple categorical feature. Nevertheless, the focus on stable correlations proved successful for this competition.
