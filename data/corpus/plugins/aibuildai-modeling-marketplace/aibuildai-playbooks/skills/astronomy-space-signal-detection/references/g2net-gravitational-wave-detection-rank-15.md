# 15th place solution

Competition: g2net-gravitational-wave-detection
Rank: #15
Source: https://www.kaggle.com/c/g2net-gravitational-wave-detection/discussion/275431

Thanks to kaggle and organizers hosting such a nice competition.
My approach is using a simple Conv2D net like below.
Sadly other complex approaches did not work well on my experiments.

[model]

* Whitening: Using average PSD. Averaging over all noise samples for each site.
* CQT Scaling with `filter_scale = 8/bins_per_octave` and (fmin, fmax)=(20, 1024).  Both abs and angle part were used.
* Augmentation
    * Horizontal/time shift
        * Pad both side and then horizontal random crop to get time shift image. -> ROC +0.002.
    * Mixup, prevent from overfitting
* GeM Fixed power 3 was better than the trainable case. -> ROC +0.001

* Scores


| net       | spec     | height | width | PB score |
| ---       | ---      | ---    | ---   | ---      |
| effnet b0 | Log STFT | 256    | 513   | 0.8760   |
| effnet b0 | CQT      | 181    | 513   | 0.8768   |
| effnet b3 | CQT      | 181    | 1024  | 0.8797   |
| effnet b3 | CQT      | 273    | 1024  | 0.8802  |

My final score is ensemble of Log STFT/CQT models.

* My code is available [here](https://github.com/Fkaneko/kaggle_g2net_gravitational_wave_detection)
