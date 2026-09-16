# Mitsui 14th Place Submission

Competition: mitsui-commodity-prediction-challenge
Rank: #14
Source: https://www.kaggle.com/c/mitsui-commodity-prediction-challenge/writeups/mitsui-14th-place-submission

# 14th Place Solution: MITSUI&CO. Commodity Prediction Challenge - Muurur

## Summary
The goal of the competition was to rank financial asset classes by Sharpe ratio over time. I established a baseline using a Ridge Regression model on log-returns early in the competition. After extensive experimentation with complex non-linear models (Boosted Trees) and Time Series approaches, I observed that these methods failed to consistently outperform the linear baseline. Consequently, the final submission relied on this simple baseline.

## Feature Selection / Engineering
* **Features Used:** I only used one feature: the logarithmic returns.
* **Reasoning:** I selected this feature because whenever I added more features, it added noise and did not improve the signal. Since I trained an individual model for each `pair_id`, adding information on the assets themselves seemed unnecessary.
* **Critical Bug:** There was a bug in the Feature Engineering on the prediction pipeline that caused returns to be generated one day shifted in the past instead of the current day. This was unintentional and worsened the performance in retrospective backtesting.

## Training Method(s)
* **Algorithm:** Scikit-learn Ridge Regression.
* **Parameters:** Standard default parameters.
* **Training Scope:** Trained on all available log-returns for each specific target pair until `date_id` 1917. I did not include the additional data from the data update.

## Interesting Findings
I conducted a rolling backtest (training on the full available dataset from `date_id` 1000, evaluating on 90 days before retraining) comparing the Ridge approach against Mean-only models.

* **Market Regimes:** In a 90-day competition window, market regime changes can make results highly variable.
* **Ridge vs. Mean:** While the Mean model is stable, the Ridge Regression model demonstrates higher variance and higher peaks. The only mean model that generated a consistent positive Sharpe ratio was calculating a mean over *all* available data.
* **Performance:** In the last year of market data, the Ridge Regression performance trended upward, outperforming the rolling mean baseline.
* **Hybrid Approaches:** Combining the mean model with Ridge regression diminished the "high highs" from the Ridge model while not offsetting the lows.

## Model Execution Time
The method is extremely simple and compute-efficient.

| Metric | Time |
| :--- | :--- |
| **Training Time** | < 5 Seconds |
| **Inference Time** | < 5 Seconds |
