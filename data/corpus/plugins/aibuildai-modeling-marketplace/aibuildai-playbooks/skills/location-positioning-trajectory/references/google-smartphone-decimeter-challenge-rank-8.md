# 8th place solution

Competition: google-smartphone-decimeter-challenge
Rank: #8
Source: https://www.kaggle.com/c/google-smartphone-decimeter-challenge/discussion/262074

##  Reproduced baseline

As my initial predictions I used approach similar to the one described [in this discussion](https://www.kaggle.com/c/google-smartphone-decimeter-challenge/discussion/238583) with one additional trick. I fitted WLS a few times, each time removing measurements with the biggest residuals. My reproduced baseline had slightly better CV score than the one provided by the host.

## Improving baseline using relative positions

This part consisted of three steps:
* I computed velocities in ENU coordinate system between consecutive timestamps using Acummulated Delta Range, pseudoragnes, sattelite positions and reproduced baseline. After that, I removed outliers and applied smoothing to predictions
*  Then I trained convolutional neural network using velocities from previous step, IMU data and baseline, which also predicted velocities in ENU coordinates. Model achieved mean absolute error around 0.25m/s .
* As a last step, I applied Weighted Least Squares to combine relative predictions with data from derived files. For each timestamp it solved the following optimization problem

$$  argmin_{x, b_{-k}, ..., b_k}  \sum_{i=-k}^k \sum_{j=1}^{n_i} w_{ij} (|| s_{ij} - \Delta x_i - x|| + b_i -p_{ij})^2 $$

where 
* \\( k \\) - parameter controlling size of the window. 
* \\( x \\) - phone position at timestamp \\( t \\)
* \\( b_i \\) - phone clock bias at timestamp \\( t + i \\)
* \\( n_i \\) - number of pseudorange measurements at timestamp \\( t + i \\)
* \\( w_{ij} \\) - inverse of \\( j \\)-th pseudorange measurement uncertainty multiplied by \\( 1 + 0.25|i| \\) 
* \\( s_{ij} \\) - \\( j \\)-th satellite position at timestamp \\( t + i \\) 
* \\( \Delta x_{i} \\) - model's prediction of relative position between timestamp  \\( t + i \\) and \\( t \\) 
* \\( p_{ij} \\) - corrected \\( j \\)-th pseudorange measurement at timestamp \\( t + i \\)

Parameter \\( k = 10 \\) worked best for me, so the window size was 21. This model had 24 unknowns - 3 for position and 1 for clock bias for each timestamp.  It had 6 times more parameters than WLS for single timestamp, but 21 times more measurements. The downside was, that any errors in relative positions affected performance of this solution.

## Improving baseline in downtown area

As all the road segments from downtown areas in the test data were also present in ground truth data, I applied the following algorithm for each timestamp:
* select all points from ground truth file closer to the baseline prediction than 30 meters - let's call them candidate points
* for each candidate point \\( x_j \\) estimate clock bias: \\( b_j = \mathrm{median}_i ( || s_i - x_j || - p_i ) \\)
* find \\( k \\) points with the lowest values of 
$$  \sum_{i=1}^{n} w_i | || s_i - x_j || - b_i - p_i | $$
and return their medoid.

It was important to use median instead of mean to compute clock bias and to use absolute value of residuals instead of squares, because it helped with very noisy measurements in downtown area.


## Postprocessing

After that, I applied the following steps to improve predictions:
* Error correction model - it predicted the difference between current predictions and ground truth. It was simple 1D Conv Neural Network. Features for this model were based on current predictions and imu.
* Aggregating predictions when speed is zero.
* Aggregating predictions from different phones.
