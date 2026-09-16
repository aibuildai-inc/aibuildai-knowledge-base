# 18th place solution (Moro & taksai)

Competition: indoor-location-navigation
Rank: #18
Source: https://www.kaggle.com/c/indoor-location-navigation/discussion/240116

Thank you to the organizers for the fun competition and everyone who participated. And thank you to my teammate ( taksai @tsaito21219 ).
I share our team's solution.

# summary
- train xy-model with only wifi-data
- train delta-model with sensor-data for cost-minimization
- postprocess repeatly 20times

# 1. preprocess
we make 4 dataset for the two model described later.
|  dataset   |describe  |
| --- | --- |
| (a)bssid-ranking     | list of bssids arranged in descending order of rssi. (common to buildings) |
| (b)bssid-rssi-matrix| bssid as a columns and rssi as data (for each building).  Fill missing-value in -120.  Drop rssi-value over 10000ms from last-seen-timestamp. |
| (c)sensor-data(aggregate) |  aggregate data between wifi. (1 record for 1 target) |
| (d)sensor-data(sampling) | sampling raw data between wifi every 100ms (N record for 1 target).  Padding to a fixed length. |

# 2. models
we make 2 type of model. First is xy-model which predict position of waypoint, Second is delta-model which predict distance between two waypoints.

## (1) xy-model
- GBDT (lightgbm)  with dataset-(b): each site
- MLP (keras) with dataset-(a): 1model
- MLP (keras) with dataset-(b): each site  -> LB=6.5 (MLP no postprocess)

## (2) delta-model
- This model is used for cost-minimization(postprocess). Since the delta calculated using the github function  has a little error, the error is reduced by creating a prediction model. 
  - error(mean of sum of squared error): 13.1(use github) -> 4.48(our model)
  - MAE(mean over x and y of absolute error) : 1.69(use github) -> 1.08(our model) 
- MLP (keras) with dataset-(c): Dense(128) > Dense(256) > Dense(128) > Dense(64) > Dense(2)
- 1d-cnn (keras) with dataset-(d):  Conv1D(filters=32,kernel_size=5,padding=2) > Conv1D(64,5,2) > Conv1D(128,3,2) > Conv1D(256,3,2) > Conv1D(512,2,2) > GlobalMaxPool1D > Dense(64) > Dense(2)

# 3. postprocess
We customized postprocessing based on some useful public kernel.
- 1st step: (LB=6.5 -> 3.0)
1) ensemble: weighted averaged the predicted value of some xy-models. The same applies to delta-model.
2) tune xy for leakage considering device-id
3) cost-minimization: use delta calculated by delta-model instead of github function.
4) snap-to-grid
5) repeat 2)-4) 20times while adjusting the threshold of snap-to-grid. Move to the grid little by little.   threshold: 2 (1-5 times) > 3 (6-10 times)> 4 (11-20 times)
6) execute 2) again
- 2nd step: (LB=3.0 -> 2.75)
1) get output(xy) of 1st step about some patterns(ensemble weight etc.)
2) execute 1st step again with 1)

# Not work
- use data of other site(over 200). I think it is efficient for delta-model, but don't work.
- train one model which predict both xy and delta (use RNN and transformer)
- learn multiple waypoints of a path together

we didn't make floor-model. Our team' floor-loss is big about private dataset. Big mistake... So private score is worse than public.
Although there were many ideas for this competition, most of them were ineffective and it was a very difficult competition.I almost gave up on the way, but since I came up with the delta-model a week ago, the score has risen by more than 1m. I'm glad I didn't give up.

Thank you to my teammate. 
I'd like to get the gold medal next time.
