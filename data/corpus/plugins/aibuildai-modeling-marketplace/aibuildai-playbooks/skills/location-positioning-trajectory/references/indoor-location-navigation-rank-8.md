# 8th place solution

Competition: indoor-location-navigation
Rank: #8
Source: https://www.kaggle.com/c/indoor-location-navigation/discussion/240151

# Absolute position model

My base model used wifi and beacon events to predict absolute position in a single timestamp. It used 50 events from 2 closest blocks of wifi events with hightest values of rssi and 10 closest beacon events. Events were sorted by `rssi`.

I trained one model on all sites. I used two bidirectional LSTMs - one on embeddings of wifi bssids + some numerical features related to each wifi event and one on embeddings of beacon_id + some numerical features related to each beacon event. Model also used embeddings of concatenated `site` and `floor`.

I think that the biggest improvement in this model was after I started interpolating waypoints in the training dataset. My best model was trained on waypoints interpolated every 1 second. 

# Relative position model

I also trained model, which predicted relative position of a smartphone based on sensor data. It predicted two numbers (deltaX, deltaY) based on 1 second of sensor data - 50 timestamps. I used Kalman Smoother to get targets for this model. It was a simple 1D-CNN model based on raw events.

Predictions from this model were interpolated, to approximate differences between positions in two consecutive waypoints. It achieved ~2.0 RMSE. 

# Postprocessing

The first step of my postprocessing was  inspired by @saitodevel01's great notebook: [indoor - Post-processing by Cost Minimization](https://www.kaggle.com/saitodevel01/indoor-post-processing-by-cost-minimization). 

I assummed that models' predictions have the following distributions

$$(\hat{X_i}, \hat{Y_i}) | (X_i, Y_i) \sim N_2((X_i, Y_i), \sigma_1^2 I_2)$$
and
$$(\hat{\Delta X_i}, \hat{\Delta Y_i}) | (X_{i+1}, Y_{i+1}, X_i, Y_i)  \sim N_2( (X_{i+1}, Y_{i+1}) - (X_i, Y_i), \sigma_2^2 I_2) $$
and all pairs \\( (X_i, Y_i) \\) are uniformly distributed in the corridor of the building, where

\\( (X_i, Y_i) \\)  is a true position at timestamp \\( i \\) , \\( (\hat{X_i}, \hat{Y_i}) \\) is a prediction of absolute model, \\(  (\hat{\Delta X_i}, \hat{\Delta Y_i}) \\) is a prediction of relative model and \\( \sigma_1, \sigma_2  \\) are RMSEs of absolute and relative models respectively.

After some math, it turns out that vector of \\( (X_1, Y_1, ..., X_n, Y_n) \\) has truncated multivariate normal distribution with easy to calculate mean and covariance matrix. It is truncated in such a way, that each pair \\( (X_i, Y_i) \\) lies in the corridor of the building.

I implemented iterative method of sampling paths from this distribution similar to Gibbs sampling. In each iteration it samples pair \\( (X_i, Y_i) \\) using untruncated normal distribution  conditioned on all the other pairs sampled so far until it samples a point inside the corridor or until it reaches given limit of trials. For each path I draw 500 samples and average last 200 samples. 

This method also allowed to use device id leak, by setting \\( \sigma_1 \\) to some small value for leaked timestamps.

This method without truncating to corridors had similar performance to Cost Minimiztion notebook  - in fact I think they are almost equivalent. Adding map information improved my public LB score by ~0.3.

The second step of my postprocessing was standard Snap to Grid, without any additional waypoints.

# Metrics

| method | private LB | public LB |
| --- | --- | --- |
| best absolute model |5.36934 |4.90242 |
| ensemble of 4 absolute models |5.30067 | 4.88062|
| above + device id leak + sampling postprocessing |3.33887  |  3.00037|
| above + snap to grid |3.07282 | 2.64846 |
