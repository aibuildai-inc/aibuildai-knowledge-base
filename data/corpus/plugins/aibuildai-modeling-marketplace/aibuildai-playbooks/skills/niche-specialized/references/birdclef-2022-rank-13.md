# Few words about 14th place solution

Competition: birdclef-2022
Rank: #13
Source: https://www.kaggle.com/c/birdclef-2022/discussion/326990

Thanks to everybody who organized this competition. Congratulations to all who received medals.
Seems that results of this competition aren't make a lot of sense because the host solution is the best.

Few words about my trying, hope you find it interesting or even useful.

**Metrics**
I start with macro Fb-score which seems are suitable for case of noisy labels. I chose quite big beta - 16 to boost recall.
After reading some comments of hosts about the metric I switched to:
`M = (1/N) * (1/21) * ∑∑( (TP/(TP+FN) + TN/(TN+FP))/2 )`

For any amount of training classes metric was calculated just over scored:
`y_true = tf.gather(y_true, indices=self.mask, axis=1)`
`y_pred = tf.gather(y_pred, indices=self.mask, axis=1)`

**External data**
2 recent recordings from Hawaii which can be found on Xeno-Canto.
ff1010bird records without birdcalls to mix with.

**Input**
As input pre-computed mel-spectrogram (mainly) or PCEN (in few cases) were used. Both calculated by Librosa with default parameters, nmels=128. So for 5s shape is 128x313.
For mel-spectrogram primitive denoising strategy was used to reduce stationary noise over whole file. On inference same denoising approach was applied but on 5s chunk level instead whole file.

In case of mel-spec, after augmentations and before mixup it was log-scaled and normed from 0 to 1.
In case of PCEN after tiny augmentations and before mixup it was normed from 0 to 1.

*Idea to beat noisy labels a little bit (was implemented for few training attempts):*
Drop time bins with low std (can be done for PCEN or log-melspectrogram):

`std_over_f = pcen.std(axis=0)`
`pcen_f = pcen[:, std_over_f > std_over_f.mean()]`

Some mask smoothing can be applied before resampling by scipy.ndimage.convolve.

**Approaches**
2 different approaches were implemented:
1.  Multiclass classifier for all (152) species. CategoricalCrossentropy loss with label_smoothing=0.1 was used. Sigmoid is used as activation function of output layer (instead of softmax). On inference predictions were normalized from 0 to 1 (divided by max value).
2. Multilabel classifier for chosen species. In addition to scored labels the species which have high overlap with scored (in secondary labels) or which have big amount of recordings  are chosen. Records for unchosen species are used for random mixing with initial input.
tf.nn.weighted_cross_entropy_with_logits with added label smoothing was used (weight=4.0, label_smoothing=0.1).

ff1010bird dataset was also tried as initial input as nocall class in 1st approach and as absence of all classes in the 2nd instead of use it for mixing.

Mainly the efficientnets (B0, B2, B3) were used with noisy-student weights, also seresnet101 was trained for mel-spec multiclass classifier.
Instead of simple global pooling in most attempts AutoPool over one axis followed by mean pooling was used.

**Augmentations**
1. Random crop of 5s window inside whole file (or restricted part in case of oversample).
2. Time stretching and compression by resize over time axis.
3. Random padding if file shorter than 5 s.
4. Random low pass filtering approximately done directly on mel-spec.
5. Mix with nobird records from ff1010bird.
6. Mix over time: up to five consecutive 5s parts can be taken and mixed with random gains. 
7. Mixup: beta=2, alpha=3*beta. Mixing lambda was forced to be >=0.5 for scored label in case of mixing scored with non-scored.

**Thresholds**
Trained models were evaluated over whole files from validation set and best thresholds were estimated. Obtained threshold for different approaches were averaged and tuned by knowledge about amount of data per bird in dataset.
Min value of 0.25 was used for 7 rare classes and just two species (skylar and houfin) have thresholds more than 0.5.

**Ensemble**
At first average predictions are estimated per each approach (mel-spec multiclass, pcen multiclass, mel-spec mixin multilabel) and then final averaging was done with coefficients selected from LB results.

**Post-processing**
For reduce FP and find out nocall parts few checks were implemented:
- std behavior for mel-spec and pcen;
- mean/max relations between whole file and current chunk;
- check scored labels max prediction over file.

#standwithukraine
#stoprussianaggression
