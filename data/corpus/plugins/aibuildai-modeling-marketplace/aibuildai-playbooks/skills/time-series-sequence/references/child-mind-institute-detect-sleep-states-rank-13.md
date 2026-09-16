# 13th place solution [team summary]

Competition: child-mind-institute-detect-sleep-states
Rank: #13
Source: https://www.kaggle.com/c/child-mind-institute-detect-sleep-states/discussion/459703

First of all, I would like to thank everyone involved in organizing this competition, all the participants, and my teammates @ktakita @stgkrtua @copasta @masatomatsui 
In this topic, I write up the summary of our team solution. The details of these models will be written by each member.
[13th place solution [K.T. part]](https://www.kaggle.com/competitions/child-mind-institute-detect-sleep-states/discussion/460499)

## Summary
- multi task learning (sleep state binary prediction & onset/wakeup event prediction)
- ensemble (9 model)
- duplicate flag feature
- multi post processing
.png?generation=1701862054954560&alt=media)

## Model
simple average ensemble by following 9 models
duration and downsampling_rate are tuned by each models
Our models is based tubo's public code. Many Thanks!

- kuto
  - 2D UNet (cv:0.763)
      - backbone: efficientnet-b3
      - duration: 5760
      - down sample: 2
  - 1D LSTM (cv: 0.786)
      - feature_extractor: 1DCNN+Wavelet transform
      - decoder: 1DCNN+LSTM
      - duration: 17280
      - down sample: 4

- copasta
  - 2D UNet (cv: 0.786)
      - duration: 17280
      - down sample: 6
  - 2D UNet (cv: 0.780)
      - backbone: EfficientNetV2-S
      - duration: 17280
      - down sample: 6
  - CenterNet (cv:0.788)
      - duration: 17280
      - down sample: 6

- K.T
  - 1D UNet (cv: 0.778)
      - Network: 4 Encoder and 4 Decoder with SE Block
      - duration: 11440
      - down sample 2
  - 1D UNet (cv: 0.801)
      - Network: 4 Encoder and 4 Decoder with SE Block
      - duration: 17280
      - down sample 4

- toppo
  - 1D UNet (cv: 0.774)
      - feature_extractor:Wavenet
      - decoder:LSTM
      - duration: 11440
      - down sample 4
  - 1D UNet (cv: 0.765)
      - feature_extractor:Wavenet
      - decoder:LSTM
      - duration: 5760
      - down sample 2

The following techniques were effective in some models.
- add L1Loss for sleep state prediction diff
- warmup
- negative sampling (bg_sampling_rate greater than 0.5)

## Dataset
features used by all models
- anglez, enmo
- hour(sin, cos transform)
- duplicate flag feature (**important**)

features used by some models
- anglez, enmo diff
- anglez, enmo lead

### duplicate flag feature
Some non-wear cases were filled with artificial data. These were created by duplicating the wave in 15-minute steps.
Therefore, we added a flag as a feature to determine if each step is a duplicated wave. This was effective for all models and improved CV and LB by about +0.005~+0.01.
Adding to the model as feature tended to improve it more than removing artificial data by post process.

## Post processing(pp)
apply following post processing after scipy.signal.find_peak.
1 and 2 pp are more important.

1. 12step(1 min) unit based pp
2. tolerance based pp
3. remove wakeup event at the beginning of each series
4. remove non pair event
5. score decay at the ending of each series


### pp details
- 12 step(1min) unit based pp (cv + 0.003)

Metric is evaluated in units of 1min (12 steps), and it is wasteful to use a step that is a multiple of 12 as the predicted value. Therefore, if the predicted step is a multiple of 12, the step is shifted by +-1.

- tolerance based pp (cv+ 0.005)

post process to bring predicted events in tolerance 12-36 within tolerance 12.
Place the score-decayed prediction, +-23 steps away from the high peak prediction(score > 0.2).
