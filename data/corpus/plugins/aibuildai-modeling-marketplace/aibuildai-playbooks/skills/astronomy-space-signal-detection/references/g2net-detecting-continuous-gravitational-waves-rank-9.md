# 9th Place Solution; Simple CNN Approach

Competition: g2net-detecting-continuous-gravitational-waves
Rank: #9
Source: https://www.kaggle.com/c/g2net-detecting-continuous-gravitational-waves/discussion/375897

Congrats to all prize and medal winners!
Because the top teams got very high scores, I look forward to learning what the magic was! Anyway, I briefly introduce my simple CNN-based solution here.



## Training data

Generating training data is very important in this competition because the given training data consists of only simulated (stationary gaussian) noises while test data contains real noises. In order to reflect the test data, I generated two types of noises; stationary noise and time-varying noise.
- Stationary noise is simply drawn from the Gaussian distribution, whose mean and std are estimated from training data.
- Time-varying noise is also drawn from the Gaussian distribution but its mean and std are vary with time. These parameters are calculated from test images.
- In addition, to simulate real noise, random walk and line noise is added to time-varying noise.
- Then, signal is inserted with a probability of 0.5.
- Finally, by deleting data at multiple timestamps, timestamp gaps in train and test data are reproduced (for both stationary noise and time-varying noise).

The key point here is that all training data is generated online except signals. Only signals (without noise) are created with pyfstat beforehand.



## Model
I trained a UNet model to predict signal positions in time-frequency domain in addition to predicting signal existence for better supervision.
