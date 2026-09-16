# 9th place summary

Competition: birdclef-2022
Rank: #9
Source: https://www.kaggle.com/c/birdclef-2022/discussion/326968

Thanks to Cornell Lab of Ornithology and Kaggle for hosting this interesting competition. I also would like to thank my teammates @truonghoang and @gwanghan for the great collaborative teamwork during the competition.

Our solution is an ensemble of 1 SED model and 1 melspectrogram classification model which score 0.76 and 0.77 respectively in private LB. ensemble gave us 0.79.

## SED
- Backbone: tf_efficientnet_b5_ns.
- Train only on the first 5s
- Inference 5s
- Since the backbone is powerful, we use quite a lot augmentation in the hope that it will help closing the domain gap between training data and the test data. The augmentation include audio augmentation, melspec augmentation, mixup 2-3 samples and cutout.

## Melspectrogram classification
- Backbone: resnest50d_1s4x24d
- Train on 7s random crop.
- Inference 7s
- Augmentation: we use simple augmentation
```
    NoiseInjection(max_noise_level=0.04, sr=SAMPLE_RATE),
    PitchShift(max_range=3, sr=SAMPLE_RATE),
    RandomVolume(limit=4),
```
- Training: 
  - Round 1: We train several models and ensemble to create an oof for each 7s crop.
  - Round 2: We conbine groundtruth label and oof to modify the target of each 7s, then train on the modified target.
  ```
  if oof_prob[primary_bird]>0.5:
      target[primary_bird] = 1.0
  elif prob[primary_bird] > 0.1:
      target[primary_bird] = 0.9975
  elif prob[primary_bird] > 0.01:
      target[primary_bird] = 0.5
  else:
      target[primary_bird] = 0.2
      
  if oof_prob[secondary_bird]>0.5:
      target[secondary_bird] = 0.9975
  elif oof_prob[secondary_bird] > 0.1:
      target[secondary_bird] = 0.8
  else:
      target[secondary_bird] = 0.0025
  ```
## Ensemble
 We found that the public lb score is very sensitive to threshold, the optimal threshold for each model are very different, so doing weight average does not bring much improvement in our ensemble. We end up multiplying their probability and set the top 33% highest confidence score in the test set as True, the rest is False.

P/S: Congrats my hard-working teammate @truonghoang on becomming competition master
