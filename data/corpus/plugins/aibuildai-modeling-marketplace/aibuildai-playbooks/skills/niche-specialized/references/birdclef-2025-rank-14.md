# 14th place solution

Competition: birdclef-2025
Rank: #14
Source: https://www.kaggle.com/c/birdclef-2025/discussion/583344

First of all, we’d like to thank the hosts and Kaggle for organizing this competition.

Congratulations to @xyzdivergence on achieving Kaggle Competitions Grandmaster—well deserved!

### Model & Training

We used a SED architecture and trained on random 10-second audio segments. For the final ensemble, we relied solely on tf_efficientnetv2_m.in21k, trained with slightly varied configurations.

The best private score for a single SED model was 0.922 (not the selected submission), and 0.894 for a CNN model using the same backbone.

Our training pipeline consisted of several stages:

* Stage 1: Pretraining on the training audio with true labels.

* Stage 2: Knowledge distillation using both train audio and train soundscapes. We combined average pseudo labels across the full audio (with a 1-second stride, weighted at 0.3) and chunk-level pseudo labels from the teacher model (10-second chunks, weighted at 0.7).

* We performed several rounds of distillation, selecting the best-performing teacher from the previous round based on leaderboard improvements.

### Features

Melspectrogram settings:

```
sample_rate: 32000
mel_bins: 128
fmin: 40
fmax: 15000
nfft: 1024
hop_length: 512
```

Augmentations:
```
On waveform: sumix (p=1)
On spectrogram: mixup (p=1), 3 time/frequency masks (p=0.5), horizontal flip (p=0.5), and random erasing (p=0.5)
```

### Final Submission

The final submission was a simple average of three tf_efficientnetv2_m.in21k checkpoints, followed by smoothing using neighboring clips with weights of 0.1, 0.8, and 0.1.

To speed up inference, all models were converted to OpenVINO.
