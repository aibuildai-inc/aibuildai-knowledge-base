# 8th Place solution

Competition: rfcx-species-audio-detection
Rank: #8
Source: https://www.kaggle.com/c/rfcx-species-audio-detection/discussion/220803

Thank you for opening this competition
Also the notebooks and discussions have helped me a lot, thank you all!

My solution was similar to Beluga & Peter in 7th Place.
@https://www.kaggle.com/c/rfcx-species-audio-detection/discussion/220443.
- Multi-class multi-label problem
- Data cleaning train_data with hand labels
- SED model

~~In my case, Pseudo labeling did not work well, so I did not use it.~~

**[hand labels]**
 While observing the data from t_min and t_max in the given train__tp.csv, I found that there are many kinds of bird calls mixed together.
 So I decided to treat it as a multi-class multi-label problem.
 It was also mentioned in the discussion that the test labels were eventually carefully labeled by humans.
 The TPs in the given t_min, t_max range are all easy to understand, but there are many TPs in the 60s clip that are difficult to understand and not labeled.
 I thought it would be better to label them carefully by myself to make the condition as close to test as possible in case such incomprehensible calls are also labeled in test.
And I was thinking of doing Pseudo labeling after the accuracy of the model improves.

 I trimmed the 5s~ around t_min and t_max in train__tp.csv.
 Hand labels took about a week.
 As a result, a total of 2428 clips and 5s chunks were used as train_data.
 The distribution of the train_data classes looks like this
(I couldn't upload the image, so I'll post it later)

class nb
s3 1257
s12 520
s18 512
:
s16 100
s17 100
s6 97

 I can see that there is a label imbalance, especially for s3, s12, and s18, because their labels co-occur among the other classes of clips.

 In particular, s3 is dominant, so it tends to output high probability, while a few classes output low probability, so i thought this is a bad problem for this evaluation index.
Therefore, in order to achieve a more balanced distribution, I oversampled the minority classes and undersampled the majority classes.
 However, the LB became worse.
 Looking back, I didn't think of approaching the test distribution, as Chris pointed out.
https://www.kaggle.com/c/rfcx-species-audio-detection/discussion/220389

Eventually, i ensembled 15 models with various losses(FocalLoss,FbetaLoss+FocalLoss,ResampleLoss).

**[trainning]**
example single model 
PANNsDense161 (public_LB 0.95548, private_LB 0.96300)

I also tried EfficientNet_b0, Dense121, etc., but Dense161 worked well.

  train_data(sr=48000,5s)
  window_size=2048,hop_size=512,mel_bins=256
  MultilabelStratifiedKFold 5fold
  BCEFocalLoss(α=0.25,β=2)
  GradualWarmupScheduler,CosineAnnealingLR(lr = 0.001,multiplier=10,epo35)

**Augmentation**
  GaussianNoise(p=0.5)
  GaussianSNR(p=0.5)
  FrequencyMask(min_frequency_band=0.0, max_frequency_band=0.2, p=0.3)
  TimeMask(min_band_part=0.0, max_band_part=0.2, p=0.8)
  PitchShift(min_semitones=-0.5, max_semitones=0.5, p=0.1)
  Shift(p=0.1)
  Gain(p=0.2)

**[inference]**
  stride=1
  framewise_output max
  No TTA (I used it in the final ensemble model)

Finally, I've uploaded the train_data_wav (sr=48000) and csv that I used. 
https://www.kaggle.com/shinoda18/rainforest-data
