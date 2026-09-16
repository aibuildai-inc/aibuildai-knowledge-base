# 4th place solution (5th public LB)

Competition: rfcx-species-audio-detection
Rank: #4
Source: https://www.kaggle.com/c/rfcx-species-audio-detection/discussion/220342

First of all, I'd like to thank my teammates, Rainforest Connection and Kaggle for this interesting and tricky challenge!

The major issue in this competition was obviously the labelling quality. True- and False-Positives audios contain lots of unlabelled regions that adds too much noise for the models. As a consequence, till the very end of the competition we haven't managed to establish a reliable local validation strategy and were mostly relying on the Public LB scores.

Moreover, labeled regions in the TP audios were balanced, i.e. each class had an equal number of labels. However, we've noticed that test predictions contain mostly the 3rd class as a top-1 probability. And with the higher percentage of the 3rd class, the LB score tends to be better. The similar situation was for some other classes (e.g. top-2 was mostly the 18th class). It gave us an idea that probably test files have completely different class distributions compared to the TP data. That's why we've applied additional multipliers for the 3rd and 18th class to artificially increase probabilities for them (naming it class balancing).

Our final solution consists of 3 stages.

## 1st Stage
* Data: only TP labels on 26 classes (for each song_type).
* Models: SED-classifiers (EfficientNet-B1 and EfficientNet-B3)
* Cropping strategy: Random crops around TP regions
* Loss: BCE
* Augmentations: spectrogram augmentations (SpecAugment, Noise) and CutMix: cutting the TP regions and pasting them into the random time regions in the other TP and FP audios.
* Public LB score: 0.909 -> 940 (after balancing)
* Private LB score: 0.915 -> 0.938

## 2nd Stage
Taking the models from the 1st Stage we've made a set of pseudolabels for TP (OOF), FP and test data. The pseudolabels have been generated using the SED framewise output. At this point, audio files have much more labeled regions compared to the initial TP data. And on this stage models are being trained on the pseudolabels only. We've applied two approaches:

##### SED-classification
* Data: TP pseudolabels  +  random 2000 samples from FP pseudolabels for each fold. Use soft labels (0.9) for the pseudolabels
* Models: SED-classifiers (EfficientNet-B0, EfficientNet-B1, MobileNetV2, DenseNet121)
* Cropping strategy: Random 5 seconds crops around pseudolabeled regions
* Loss: modified LSEP loss
* Augmentations: raw audio augmentations, such as: GaussianNoiseSNR, PinkNoiseSNR, PitchShift, TimeShift, VolumeControl 
* TTA: 6 different crop sizes are used during the inference: 2, 5, 10, 20, 30 and 60 seconds
* Best single model (5 fold) public LB score: 0.957 (after balancing)
* Private LB score: 0.963

##### Usual classification
* Data: TP + FP pseudolabels. Pre-train models on the test pseudolabels
* Models: Usual classifiers (EfficientNet-B1, ResNet34, SE-ResNeXt50)
* Cropping strategy: Random crops around pseudolabeled regions
* Loss: BCE
* Augmentations: spectrogram augmentations (SpecAugment, Noise) and CutMix
* Best single model (5 fold) public LB score: 0.952 (after balancing)
* Private LB score: 0.959

## 3rd Stage
Taking the overall ensemble from the 2nd Stage allows to get the Public LB score of 0.965 (Private LB: 0.969). To achieve our best 0.969 Public LB (Private LB: 0.971) we're applying single class semantic segmentation models for 3rd, 11th and 18th classes (other classes didn't give any score improvements on the Public LB).

The segmentation polish is done in the following manner:
`class_score = class_score * (1 + 0.1 * num_instances) if num_instances > 0 else  class_score * 0.9`, where `num_instances` is the number of instances predicted by the semantic segmentation model for each recording.

## What didn`t work 
1. PANN pretrained weights (or other audio pretrained models) - imagenet performs best
2. Using "fat" encoders
3. Focal loss with soft penalty (But as we see It works for the other participants)
4. Multiclass segmentation
5. Raw audio classification with 1d convolutions
