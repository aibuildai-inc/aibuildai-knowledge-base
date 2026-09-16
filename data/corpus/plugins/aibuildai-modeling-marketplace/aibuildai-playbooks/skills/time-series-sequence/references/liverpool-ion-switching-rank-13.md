# 13th place solution

Competition: liverpool-ion-switching
Rank: #13
Source: https://www.kaggle.com/c/liverpool-ion-switching/discussion/153863

Congratulations to the winners, during last week the competition had been very hard!

The most difficult part was to find a good way to clean the signal micro-drifting (which is different across batches and for train and test) .  To do that, I fit a gaussian mixture to batches of 100k points, and linearly ‘lift’ the signal to adapt the gaussians means to a set precalculated fixed signal levels, common for train and test data. The gaussian mixture fit also give an unsupervised estimate for open channel probability for both train and test.

Using that probability as guess for the open channels is easy to spot the 50Hz  and harmonics AC component using FFT and remove it surgically. ( see details of the calculations in this notebook  [https://www.kaggle.com/vicensgaitan/1-remove-drift-ac](https://www.kaggle.com/vicensgaitan/1-remove-drift-ac)

Firstly I try lightgbm for modeling, so some feature engineering was needed.
I build some time-symmetrical rolling features (averaging for the signal and the time reversed signal) . Tree based models using these  features and probabilities from the gaussian mixture scored around 0.942 in public LB

The game changer is the WaveNet architecture. Using the same set of variables with a plain WaveNet ( no LSTM, no batch normalization, no dropout, no fancy heads) using cross entropy loss, we can easily achieve  0.946 public (0.945 private) .  A single model can do almost gold 
[https://www.kaggle.com/vicensgaitan/2-wavenet-swa](https://www.kaggle.com/vicensgaitan/2-wavenet-swa)

My 13th position is a bagging of 5 models with different seeds and learning  rates.

I miss the 5+5 structure of the 10-channel data…. Too bad

There is still one mystery: The leaderboard results (public and private) is 0.004 points better than the CV value, correcting by batch composition. It seems that test data has some ‘leak’ from the train data,  maybe  this related to the high scores obtained by the competition winners
