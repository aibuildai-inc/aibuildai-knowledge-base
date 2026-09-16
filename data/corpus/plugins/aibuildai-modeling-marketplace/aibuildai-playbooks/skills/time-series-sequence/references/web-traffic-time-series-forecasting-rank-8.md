# 8th place with Kalman filters

Competition: web-traffic-time-series-forecasting
Rank: #8
Source: https://www.kaggle.com/c/web-traffic-time-series-forecasting/discussion/43727

I also did quite well without using any neural networks, XGBoost or other modern methods. Here's an overview of my not-so-elegant but fully "classical" approach.

It's based on the observation that most of the time series are low-traffic, noisy and seemingly very unpredictable (figure 1) while some of them behave quite nicely (figure 2). My main idea was to use Kalman filters to predict well-behaved time series while falling back to a more robust median-of-medians for the bulk of the data.

I was heavily aided by the public kernels: my fall-back method was almost the same as the best public kernel: the Fibonacci median of medians, grouped by weekend vs weekdays, rounded to the nearest integer (this one https://www.kaggle.com/rshally/web-traffic-cross-valid-round-and-wk-lb-44-5).

In more detail, the solution worked as follows

 1. For spiders, always use the rounded Fibonacci median of medians but _without weekly seasonality_. This was because the vast majority of the spider data was very noisy and I assumed that bots do not care about weekends. Leaving out weekly seasonality did not have much effect in my final CV benchmarks but I decided to use the simpler method since I thought it would be more robust to possible abnormalities caused by this competition.

 2. For non-spider data, try three different smoothing parameters _s_, starting from least smoothing:
   - run a Kalman smoother _K(s)_ on the log1p-transformed data to get the smoothed mean _y_
   - compute deltas: _dy_ as `np.diff(y)`
   - compute the yearly seasonality error as `R = 0.5 * |dy1 - dy2| / (|dy1| + |dy2|)`, where _dy1_ is the last year of _dy_ and _dy2_ the second last year of _dy_ (i.e., compare the data to itself, shifted by one year, very similar to what Nathaniel Maddux seems to have done in his solution)
   - if _R_ is less than a threshold value (0.95), the _s_-smoothed data is seasonal: predict the new data with the Kalman filter _K(s)_ adding yearly seasonality from `np.cumsum(dy2)`. Finally apply `exp(x)-1` and round to the nearest integer 
   - otherwise try the next smoothing level _s_

 3. If the data is not seasonal with any smoothing level, fall back to rounded Fibonacci median of medians by weekend

I used a 8-state Kalman filter representing a local level (no trend) and weekly seasonality, parametrized by a smoothing parameter _s_ which determined the covariance matrices (process and measurement noise). Adding the yearly seasonality directly to the Kalman filter would have exploded the number of states or required special tricks so I handled that separately as described above.

I wrote my own SIMD-style vectorized implementation of the Kalman filters which allowed running them relatively fast in Python (Numpy). EDIT: These custom Kalman filter codes can be found here https://github.com/oseiskar/simdkalman

The total execution time of the final model was about 25 minutes on a 4-core i5, which was nice since that was about the time I had available for dealing with the final data. My two submissions were the same model executed on the 8/30 and 9/10 data. The former scored only 39.2.

In the final cross-validation / back-testing I used 5 different truncations of the data and ensured that the submitted model worked well for _all_ of these cases (i.e., max error instead of mean error). It had quickly became apparent to me that optimizing the public LB was the same as over-fitting for predicting January, which was quite different from predicting October or December.

Figures (red: predicted, dashed: actual)
