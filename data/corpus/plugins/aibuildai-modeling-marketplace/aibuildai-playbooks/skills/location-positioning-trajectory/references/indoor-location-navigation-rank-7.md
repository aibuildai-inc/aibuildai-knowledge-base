# 7th place solution

Competition: indoor-location-navigation
Rank: #7
Source: https://www.kaggle.com/c/indoor-location-navigation/discussion/240259

First of all, I would like to thank the host for organizing such a wonderful conference. I would also like to take this opportunity to thank all the wonderful members( @mmotoki, @ryches, @takoihiraokazu , @masatomatsui ) have worked so hard throughout the competition. Finally, I would like to congratulate for @mmotoki becoming a GM with this medal. Congratulations! I'm as happy as myself!

This is our team's solution. Have fun, everyone!

# Quick summary
Our solution is relatively simliar to some of other teams in the same range as us. Our final submission is a blend of LSTM and LGBM models trained with repeated pseudo labeling and postprocesing. Here we provide a brief description of some of the ideas that worked for us and rough estimates of the gains that we got from them.

[slide]


# Preprocessing
- Linear interpolation of coordinate at wifi steps
- Timestamp using the fix from [Jiwei’s Fix the timestamps of test data using DASK](https://www.kaggle.com/jiweiliu/fix-the-timestamps-of-test-data-using-dask)

# Model
## Floor Model
We used an MLP inspired by [Nigel’s 99% Accuracy public notebook](https://www.kaggle.com/nigelhenry/simple-99-accurate-floor-model). The idea seemed similar to target encoding where each bssid is mapped to the floor where its max rssi was observed. The MLP generalized this approach using the hard floor estimates (descried earlier), a soft floor estimates derived using a weighted average of bssid-floors where the weight comes from the softmax of rssi, a learned bssid embedding and a beacon weight derived from beacon measurements (e.g., rssi, last_seen). Our model changed the floor predictions on 11 of the paths predicted by the public notebook. Our model had 100% correct predictions on the public and private test sets. 

## Relative Position Model
We used a wavenet model for predicting the relative position between two waypoints.
Inputs are 12 Sensor data (accem gyro, ahrs, and magn). This model resulted in a mean delta positioning error of 1.68 m, and the host’s “compute_rel_positions” code results in a mean error of 2.84 m.

# Position Model
- LGBM - This model was based on the [BIZEN’s public notebook](https://www.kaggle.com/hiro5299834/wifi-features-with-lightgbm-kfold). The base LGBM model was not very effective, but it was particularly effective when pseudo labeling was used.
- LSTM - Our LSTM model used features from [Kouki’s LSTM by Keras with Unified Wi-Fi Feats notebook](https://www.kaggle.com/kokitanisaka/lstm-by-keras-with-unified-wi-fi-feats) and extra features including the delta of x,y calculated by the host’s “compute_rel_positions”. The length of the input sequence was 2*n+1 (n each before and n after the target to be predicted). We used device id to connect the paths of train and test, and created the input sequence. Our final submission used an average of two slight variants, the score before the post process was public:4.6 and private:4.9. Averaging the raw predictions of our 2 LSTM models (no pseudo labeling) reduced the error by about 0.1 (public:4.5 and private:4.8)

# Postprocessing:
Like most other teams, this step was really important for us. We used a lot of the ideas shared publicly with a few modifications and additional steps
- tomooinubushi’s[ time leakage](https://www.kaggle.com/tomooinubushi/postprocessing-based-on-leakage) (1st)
- nn cost minimization (e-toppos) - similar to the public magn cost minimization but using a blend of relative position predictions and the official code estimates
- Iterative closest point algorithm. [ICP](https://en.wikipedia.org/wiki/Iterative_closest_point) is often used for robot path planning and alignment of scanned point cloud data, but we thought it could also be used for alignment between estimated paths and grids. We used the training grids and the auto generated grids proposed by e-toppo 
- snap to corridor - Move estimated step positions outside corridor or building to be inside nearest corridor. We used the polygon data of the corridor, and snapped to the corridor using shapely's method “nearest_points”.
- snap to grid - Almost the same as [the public notebook](https://www.kaggle.com/robikscube/indoor-navigation-snap-to-grid-post-processing), but we divided the snapping into two stages, first snapping to the training grids and then snapping the rest to the auto-generated grids.
- tomooinubushi’s time leakage (2nd)
- ganchan’s [Device leakage](https://www.kaggle.com/iwatatakuya/use-leakage-considering-device-id-postprocess)

It is difficult to quantify how much each step in our postprocessing pipeline helped, but most of our gains came from the non-leak postprocessing steps. Postprocessing improved the score of our raw predictions by about as 1.0 (public:3.5 and private:3.8).

# Repeated Pseudo Labeling
This was also a really important step for us. After training our base LSTM models, we averaged their predictions, and used the postprocessed predictions as pseudo labels for LGBM. We retrained our LSTM models with the LGBM pseudo labels and repeated the process until it converged. Repeated pseudo labeling improved our score by another 1.0 (public:2.5 and private:2.8)
