# 2nd place solution

Competition: LANL-Earthquake-Prediction
Rank: #2
Source: https://www.kaggle.com/c/LANL-Earthquake-Prediction/discussion/94369#latest-564291

Update: Feature importance label fixed (5th June GMT 3 am) 

I select segments from the traning ​​data and create a private-set-like data based on the inter-earthquake times seen in the figure "Are data from p4677?"

mykper: https://www.kaggle.com/c/LANL-Earthquake-Prediction/discussion/90664#latest-535844

[quake_private_like_set]

The dashed line is 11.5 sec.

The first segment is counted twice; the private set has 8 segments. The last segment is created from the 14-sec segment and shifted up by 2 seconds, and the last 4.5 second is cut, treating that as the public part.

I use 32 features:

- standard deviation (std)
- std_nopeak: Std of data that are not part of peaks
- kurtosis_truncated: Kurtosis of data with abs(v - mean(v)) &lt; 20
- 7 peak counts: Number of peaks with hight &gt; 50, 75, ..., 200
- 5 percentiles: 95 percentile - 5 percentile, 80 - 20, 70 - 30, 60 - 40.
- trend: slope of robust linear regression to 30 sub chunks of std_truncated
- trend_error: Abs difference in the slope of RANSAC and Huber fit
- power spectrum; Fast-Fourier Transform the data and average the absolute value in 15 bins

I choose combinations such as 95 percentile - 5 percentile to avoid direct dependence on the mean; which is drifting with time. Same for the peak height; the peak height is defined as (max - min)/2.

The std_truncated (std instead of kurtosis in kurtosis_truncated) works almost as well as std_nopeak.


I randomly select 2000x1000 training chunks of length 150_000 from my private-like set, which is 1000 times the number of independent/non-overlapping chunks and put all of them into CatBoost. I do not provide CV data to the regressor; eveything ​is the training set.


I also tried to predict time since failure using all the training set and tried to stitch together with time to failure, but I was not able to do that successfully.

This is the feature importance:

[Feature importance]
