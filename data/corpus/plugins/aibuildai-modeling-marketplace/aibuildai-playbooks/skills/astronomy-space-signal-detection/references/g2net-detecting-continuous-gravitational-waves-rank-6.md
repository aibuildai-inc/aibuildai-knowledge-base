# 6th place solution (Simulated Annealing Approach)

Competition: g2net-detecting-continuous-gravitational-waves
Rank: #6
Source: https://www.kaggle.com/c/g2net-detecting-continuous-gravitational-waves/discussion/375923

First of all, thanks to the host and congratulations to the winners.
I'm so glad to win my first solo gold medal in this competition!
My solution is based on [Simulated annealing](https://en.wikipedia.org/wiki/Simulated_annealing) (SA), a technique often used in optimization problems. It is NOT based on machine learning/deep learning/data generation.

### Preprocess

The following process was performed for each of H1 and L1 and added together to produce a 360*256 image.
- Calculate the square of the amplitude
- Removal of horizontal line noise (only real noise)
	- Calculate the sum for each row and create a list of 360 elements. Then take a difference of this list. If there is a series of large plus value and large minus value with short intervals, it is assumed that there is horizontal line noise. Then the values of these rows is replaced by the average of the each column. This operation is repeated until it can no longer be done.
	- This removes almost all of the real noise and improves the score by about 0.005.
        - Some example images before and after horizontal line noise removal are as follows.



- Removal of vertical stripe noise
	- Subtract mean for each column, divide by std^2, and correct so that the sum of std of all columns is a constant.
	- By dividing by the square of std instead of std, the columns with larger original variance will have smaller final variance. The idea is to reduce the contribution of columns with large original variance because they cannot be relied upon. The score increases slightly by this method.
- Filling the time gap
	- Calculate the difference of the timestamps as d and repeat inserting zero-filled columns round(d / 1800 - 1) times.
- Average in the time direction and reduce columns size to 256

### Wave Detection with SA

I will explain the essential part of my solution: Simulated annealing (SA).
First, assume that the signal is perfectly sinusoidal and follows the following equation

f(x) = Asin(vx+θ) + h

where A is the amplitude, v is the frequency, θ is the phase, and h is the position in the frequency direction.
The search range for each parameter is set as follows based on the experimental results.
- A: [-100, 100].
- v: fixed at 0.31 (fixing the frequency slightly improved the score)
- θ: any real number
- h: any real number

Then we want to search the parameters that gives the clearest waveform.
To do this, I used SA, as following procedures.

- The following operation is repeated tens of thousands of times with decreasing the temperature.
  - Changes the above parameters slightly, and calculates the sum of the squares of the amplitudes along the waveform as the score (penalty if it refers outside the image). If the score is greater than the last score, accepts the change. Even if the score is smaller than the last score,  accepts the change probabilistically, depending on the score difference and the annealing temperature. Otherwise rejects the change. 
- In addition, to prevent getting a score of local optimum, the above whole process is repeated about several hundred times, and the kth percentile value (k chosen from around 90~100) is used as the final score.
- It takes several hours to a day (depending on the number of iterations) on my CPU (Core i9-9900K) to calculate the scores of all the test data.
    - Postprocessing and submitting this can achieve LB score above 0.79 ,with a few hours calculation.

Some example images of detected waveforms for some difficult test cases are as follows.




### Postprocess

I performed a number of SAs with various numbers of iterations, parameter search ranges, wave thickness, etc., and calculated the ensemble score by weighted averaging.
I gave up on calculating the CV score, and adjusted the weights so that Public LB would be higher.
In addition, the real noise data tended to score slightly higher than the generated noise data (due to its tendency to overfit against noise). Therefore, I first converted the final score to a rank, and then divided the rank value by about 1.1 for real noise data only, improving the score by several points.
