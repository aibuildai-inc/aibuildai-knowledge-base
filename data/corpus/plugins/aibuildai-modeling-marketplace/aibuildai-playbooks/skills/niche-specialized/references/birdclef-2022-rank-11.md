# 11th place solution

Competition: birdclef-2022
Rank: #11
Source: https://www.kaggle.com/c/birdclef-2022/discussion/326979

Closing time! Very painful to see us finish 1 spot shy from a gold medal, but you win some you lose some. **EDIT:** due to one team being disqualified, we managed to actually get the last gold spot!

Our final solution is an ensemble of five different models trained with different backbones or spectrogram configurations. Each of those models are heavily inspired by the [second place solution of last year](https://www.kaggle.com/competitions/birdclef-2021/discussion/243463) which was re-implemented on this dataset by @julian3833 in [this notebook](https://www.kaggle.com/code/julian3833/birdclef-21-2nd-place-model-submit-0-66).

# Phase I: tuning the threshold 

Early in the competition, we were still kind of doubting whether to commit or not, due to the lack of a true validation set. I decided to just start from the notebook mentioned above but apply the same thresholding technique as the 2nd place solution of last year in which they predicted the top-K percentile of probabilities to be True. Last year, the threshold was extremely high at around `0.9987`, this year, a threshold of `0.69` (nice) gave us a score of around 0.73-0.75 on the LB. In the end, the threshold of our ensemble was set to be `0.78`, it turned out `0.75` even worked slightly better on the private.

# Phase II: improving the pipeline

We decided to try and add different things to the pipeline in order to improve it. Most of the things we tried ended op in section 4 of this write-up unfortunately. But one thing that worked really well was simple augmentation on spectrogram-level ([SpecAugment](https://arxiv.org/abs/1904.08779)). We just masked time & frequency bands.

# Phase III: the ensemble

As mentioned, two different spectrogram configurations and different backbones were used. The spectrogram configurations were:

A:
```
cfg.window_size = 1024
cfg.hop_size = 320
cfg.sample_rate = 32000
cfg.fmin = 50
cfg.fmax = 14000
cfg.power = 2
cfg.mel_bins = 64
cfg.top_db = None
```

and

B:
```
cfg.window_size = 1024
cfg.hop_size = 512
cfg.sample_rate = 32000
cfg.fmin = 16
cfg.fmax = 16386
cfg.power = 2
cfg.mel_bins = 128
cfg.top_db = 80.0
```

The five models in our ensemble were:
* `seresnext26t_32x4d` with A and B (all scored 0.81 on public)
* `eca_nfnet_l0` on A and B (scored 0.81 on public)
* `seresnext50_32x4d` with B (scored 0.81 on public)

# Phase IV: post-processing

[One post-processing trick](https://www.kaggle.com/competitions/birdclef-2021/discussion/243343) proposed by @iafoss worked rather well here too. Here, the audio was slided by some offset (in our case 1.5 seconds forward and backward) and then aggregated as follows: `p = 0.5*p0 + 0.25*pr + 0.25*pl`. Another trick that worked marginally well is increasing the probabilities of a bird in all clips of a soundscape if we were very confident (90-percentile within that soundscape) of that bird appearing in another clip within that same soundscape.

# Things that did not work

* waveform augmentation
* adding background noise from ff1010 dataset
* including audio clips with rating <= 2
* balanced class weights as the metric was something close to macro F1
* post-processing based on co-occurrences
* using YamNET as a bird/no-bird classifier and using it to sample crops from our clips

# Our team name

When we had to think of a team name, @moeflon googled `the dumbest bird in the world` and found Kakapoo. The name of that bird contains both 💩 `poo` 💩 and  `kaka` (which is Dutch for poo) so it really was a no-brainer.

[Kakapoo bird]

# Credits

A huge thanks goes out to my teammates @moeflon @gertjandemulder @emield @jeroenvdd
