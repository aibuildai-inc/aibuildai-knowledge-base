# 7th place solution

Competition: google-smartphone-decimeter-challenge
Rank: #7
Source: https://www.kaggle.com/c/google-smartphone-decimeter-challenge/discussion/261732

First of all, I would like to thank host for organizing this competition. I participated in the competition as a soloist from start to finish, and although it was a very tough competition, but it was very meaningful as it gave me a chance to experience the interesting technology of GNSS.  

Here is my solution.  

# 1.Baseline improving
I have rebuilt the baseline based on [this notebook](https://www.kaggle.com/hyperc/gsdc-reproducing-baseline-wls-on-one-measurement).  
For isrbm, I used the median value for each phone-sat.  

## 1-1.Selecting a satellite
I improved the baseline by excluding satellites, which are a source of error, from the least-squares calculation.  

For the filter condition, I mainly used the elevation angle. Signals from satellites with low elevation angles are excluded because they are strongly affected by various errors.  

## 1-2.Carrier smoothing
Pseudorange smoothing with Acumulated Delta Range (ADR), as described in [this notebook](https://www.kaggle.com/gymf123/onepager-tip-acumulated-delta-range-adr). The original pseudorange, and the previous pseudorange + ADR Diff mixed in a certain ratio to form the final pseudorange. ADR is relative but accurate, so it can be combined with the absolute value of pseudorange to improve the accuracy. Since ADR can have an accumulated value of zero due to cycle slip, I applied carrier smoothing only when AccumulatedDeltaRangeState = 25.  

# 2.Estimation of relative position
## 2-1. Vehicle speed calculation using doppler shift
The relative velocity between the satellite and the vehicle can be determined by the frequency change of the signal (doppler shift). To determine the speed of a vehicle, we need the position of the satellite, the position of the vehicle, the speed of the satellite, the distance between the satellite and the vehicle, and the doppler shift. The position and speed of the satellite are given as data. The position of the vehicle and the distance between the satellite and the vehicle are obtained from the calculation results of Baseline improving (Need to subtract clkbias from pseudorange). The doppler shift is given as PseudorangeRateMetersPerSecond. The vehicle speed is then calculated by the least squares method using information from multiple satellites.  

The vehicle speed (and the relative position calculated from it) obtained by this method was very accurate, and was a major factor in improving the score.  

## 2-2. ML prediction (add IMU data) 
Since the relative positions obtained in 2-1 are missing in some places, I also combined IMU sensor data to create a machine learning model to supplement them. I built a prediction model in lightGBM with lag and rolling features of the IMU and vehicle velocity.

# 3. Reject outlier
There are some outliers in the baseline, which I will remove.

## 3-1. Abnormally high speeds  
Exclude points that have a very large distance from the previous and next point. 

## 3-2. Based on ground truth
Since some areas have overlapping test and train paths, I were able to use the ground truth of the train to determine the outlier. The closest distance to the ground truth data was calculated for each point, and those above the threshold were removed as outlier.  

## 3-3. Based on reference point calculated by relative position 
Since there are many test data that have paths that do not exist in train, the 3-2 method can only be used in a very limited way. To solve this problem, I created a reference point that can be used as an alternative to ground truth.  
For this, I used the relative positions calculated in 2. Starting from the coordinate point at each time, the coordinates before and after a certain time are calculated based on the accumulated relative values. By sliding this process at each point in time, a large number of estimates can be obtained at each time. The accuracy of these estimates is highly dependent on the accuracy of the absolute coordinates of the starting point. If the starting point is an outlier, the estimated value will also be an outlier, but this is not frequent and the effect can be eliminated by clipping the estimated value. We then calculated a threshold value from the mean and standard deviation of the estimated values at each point, and used it to remove the outlier values.  

# 4. Post process
## 4-1. kalman smoothing
I used [this notebook](https://www.kaggle.com/emaerthin/demonstration-of-the-kalman-filter) as is.  

## 4-2. Processing the speed0 period
As discussed in [this notebook](https://www.kaggle.com/t88take/gsdc-eda-error-when-stopping), there is a tendency for absolute coordinates to be highly scattered when the car is stopping. To solve this problem, I created a model to predict stops, and replaced the continuous periods predicted as stops with the average of those data.  

## 4-3. Cost minimization
I used the [cost_minimization notebook](https://www.kaggle.com/saitodevel01/indoor-post-processing-by-cost-minimization). Improved the accuracy of the absolute coordinates based on the relative position obtained in 2.  

## 4-4. Position shift
I used [this notebook](https://www.kaggle.com/wrrosa/gsdc-position-shift) as is.

## 4-5. Weighted phones mean
As described in [this notebook](https://www.kaggle.com/t88take/gsdc-phones-mean-prediction), this process averages the values of multiple phones in the same collection. It has been improved from the published version and changed to add weight to each phone model.  

# Area grouping
I added a little logic to the kNN introduced [here](https://www.kaggle.com/columbia2131/area-knn-prediction-train-hand-label), and implemented grouping based on the degree of path matching with the train. Each collection was divided into five groups, and the hyperparameters and order of processing were adjusted for each group.
