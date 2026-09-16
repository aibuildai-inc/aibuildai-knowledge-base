# 25th Place Silver – MITSUI&CO. Commodity Prediction Challenge Writeup

Competition: mitsui-commodity-prediction-challenge
Rank: #25
Source: https://www.kaggle.com/c/mitsui-commodity-prediction-challenge/writeups/25th-place-silver-mitsui-commodity-prediction

# 25th Place Silver – MITSUI&CO. Commodity Prediction Challenge

## Overview
This solution is based on a **group-wise ensemble of tabular models** combined with **inference-time lag blending**.  
Targets are grouped using metadata from `target_pairs.csv` (LME / JPX / US / FX), and each group is trained with a **restricted, market-relevant feature set** to reduce noise.  
The most important performance gain comes from **heavy utilization of provided label lags (1–4)** during inference.

## Target Grouping
Targets are assigned to groups according to market information extracted from the `pair` field in `target_pairs.csv`.  
Single-market and cross-market targets are handled separately, allowing each group to focus on its dominant market dynamics.  
Each group uses a different subset of features, primarily raw signals and aggregates relevant to that market.

## Feature Engineering
Features are mostly **cross-sectional** and computed per time step:
- Global statistics: mean, std, min, max, percentiles, skewness, kurtosis, IQR
- Market-level aggregates for LME, JPX, US, and FX
- High-variance raw signals (top ~150 by variance), with first-order differences
- Lightweight technical indicators on selected signals (SMA, RSI-like, Bollinger width)
- Limited interaction and ratio features

All features are cleaned for NaN/inf and filled conservatively to avoid leakage.

## Modeling
For each target, three models are trained:
- **LightGBM (GBDT)** with TimeSeriesSplit when data size permits
- **CatBoostRegressor** with moderate depth and learning rate
- **Ridge Regression** as a linear baseline / stabilizer

Predictions from these models are **uniformly averaged** at the target level.

## Inference-Time Lag Blending
During inference, the provided `label_lags_1` to `label_lags_4` are blended into the prediction:
- Strong weight on lag-1, with decreasing weights for older lags
- A small momentum term is added using recent lag-1 values

Final prediction is a weighted combination of model output and lag-based signals.

## Smoothing and Normalization
To reduce step-to-step volatility:
- Short-term EMA-style smoothing is applied using recent predictions
- Cross-sectional normalization is used to preserve relative rankings
- Extreme values are clipped to maintain stability

## Validation
Models are trained on the most recent portion of the training data.  
TimeSeriesSplit is used where applicable, and leaderboard consistency was the primary model selection criterion.

## Code
**Full implementation:** `https://www.kaggle.com/code/halilaka/notebook5da6d36740`
