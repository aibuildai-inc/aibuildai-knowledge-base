# 6th Place Solution (shimacos part)

Competition: google-smartphone-decimeter-challenge
Rank: #6
Source: https://www.kaggle.com/c/google-smartphone-decimeter-challenge/discussion/261904

First of all, I would like to thank hosts for organizing such a unique competition!
And in this competition, I’m able to become a Grand Master and my teammate @fuumin621 is able to become a master!


Here is an introduction to my part of our brief solution.
It was important for us to use AccumulatedDeltaRange properly and look at the data carefully..

# Preprocess

I predicted the anomaly (distance from ground truth) and speed from IMU data and baseline data by lightGBM.
Then, I replaced anomalies above the threshold with NULLs and linearly interpolated between them.

# Kalman smoothing

I applied two pattern Linear Kaman Smoothing.
One takes into account velocity, and the other takes into account acceleration (i.e. 4D and 6D).
Also, since the interval of the observed values was not constant, I replaced delta_t in the state transition matrix for each update step.
I tried making the observation noise larger in downtown, predicting the yaw rate, and using a nonlinear Kalman filter, but it didn't work.


# AccumulatedDeltaRange

I knew that phase observation was important, so I tried to improve the baseline using the AccumulatedDeltaRange from the satellite data, but it did not work due to lack of knowledge…
So I decided using the [GNSS Analysis app from google] (https://developer.android.com/guide/topics/sensors/gnss).
By using this tool, I was able to obtain observations using the AccumulatedDeltaRange for some of the data.
The observation data itself was difficult to use directly because it contained bias, but the relative distances obtained by taking the differences between them were very accurate like following image. (delta_adr = ADR(t) - ADR(t-1))
[image]

I used this to implement a forward hatch filter and a backward hatch filter and averaged them after applied Kaman smoothing.
```python
    prev_smooth = lat_lngs[0]
    res = [prev_smooth]
    for raw, delta_adr, delta_t in zip(
        lat_lngs[1:, :], delta_adrs[1:, :], delta_ts[1:]
    ):
        if not all(np.isnan(delta_adr)):
            smooth = 1 / M * raw + (M - 1) / M * (
                prev_smooth + delta_adr * delta_t
            )
        else:
            smooth = raw
        prev_smooth = smooth
        res.append(prev_smooth)
   res = np.array(res)
```
There is a parameter called M, so I optimized it using GroupKFold for each area. (Find the parameter that is the smallest in the train data and verify it with valid data.)
Smoothing with this hatch filter was very effective.
There were many missing delta_adr values for some phones, so we filled them with values from other phones.

# Weighted phone mean

After applied hatch filter, I weighted average latitude and longitude using the weights of phone optimized by GroupKFold as well as M.
Some of the `millisSinceGpsEpoch` had a little gap depending on the phone, so I used linear interpolation to fill it.


# Snap2grid

In the downtown area, I performed a discrete optimization using linearly interpolated ground truth.
To prevent overfitting, the ground truth in itself was not included in the target to be snapped.
The objective function is as follows, and I used greedy search to find the optimal path.
```python
cost = distance2cur + distance2prev * alpha
```

# Stay point mean

Using the predicted value of the speed of lightgbm, I took the average for points below the threshold.

# Ensemble

Using various parameters, I took the average of them and the CV was about 2.149.
Finally, after adding other members' models and post-processing, we got CV  2.0569 and LB 2.285.
In Private, I knew that there was a lot of weight on highways and trees, so I decided on the ensemble weights taking them into account.
