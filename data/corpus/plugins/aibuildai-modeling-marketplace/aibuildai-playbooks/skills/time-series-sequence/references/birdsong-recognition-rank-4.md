# 4-th place solution

Competition: birdsong-recognition
Rank: #4
Source: https://www.kaggle.com/c/birdsong-recognition/discussion/183339

Hi ,Kagglers!

It was an amazing competition! And I am happy to share training tips that helped us reach 4-th LB position

# Data Preparation

We used resampled into 32 000 sample rate and normalized audio

As most competitors, we used spectral features - Logmels

For most of my models we have added external xeno-canto datasets:
https://www.kaggle.com/rohanrao/xeno-canto-bird-recordings-extended-a-m 
https://www.kaggle.com/rohanrao/xeno-canto-bird-recordings-extended-n-z
Great thanks to @rohanrao 

Also we used RMS trimming, in order to get clips with bird calls. More details you can find in [our notebook](https://www.kaggle.com/vladimirsydor/4-th-place-solution-inference-and-training-tips?scriptVersionId=42796948)

# Feature Extraction 

As we were using pretrained CNNs, we have to support it with 3-channel input (or inplace first Conv)

Worked:
- Repeat of Logmel 3 times - good baseline option
- Use [deltas](https://pytorch.org/audio/functional.html#compute-deltas) - We have used concatenation of Logmel, 1-st order delta and 2-nd order delta. It worked the best
- Adding `secondary_labels` for training

Not Worked great:
- Time and Frequency encoding - originally used by @ddanevskyi in [Freesound](https://www.kaggle.com/c/freesound-audio-tagging-2019/discussion/97926). But it does not work well here
- Adding some more features, like Loudness and Spectral Centroid

# Validation Scheme

All our models were trained in cross-validation mode. So we had one fold for validation. Also we used `example_audio` as one more validation set. All in all we tracked such metrics:
- loss
- MaP score by one validation fold - `map_score`
- Original competition F1 metric with threshold 0.5 on test set - `f1_test_median`
-  Original competition F1 metric with best threshold on test set
- Original competition F1 metric  with threshold 0.5 on validation fold (if we use `secondary_labels`)
- Original competition F1 metric  with threshold best threshold on validation fold (if we use `secondary_labels`)

We made early stopping  and scheduling by MaP score, as it has converged the last one from all metrics

Then we took 3 best checkpoints by `f1_test_median` and averaged weights matrices for each fold - some kind of SWA. Then blend 5 models (5 folds) and evaluate on test_set (example audio). This score correlates well with LB till 0.607 point. After this point test_set was nearly useless :)

# Model

We used different EfficientNets (B3, B4, B5) pretrained on [noisy student](https://github.com/rwightman/gen-efficientnet-pytorch).

We tried some classifier heads. But 2 Layer Dropout->Liner->Relu works the best. Also 
 [Multi-Sample Dropout](https://arxiv.org/pdf/1905.09788.pdf) slightly boosts the performance and give some more stability

We tried SeResnexts but they did not work at all for us. Also we tried model proposed by @ddanevskyi [here](https://github.com/ex4sperans/freesound-classification) but it worked worse.

# Training process 

We used `Adam` optimizer and   `ReduceLROnPlateau` scheduler and `BCEwithLogits` loss

Augmentations really boosted performance (~2%). We listened to example audio and tried to choose such augmentations, that can shift our train set to example audio:
- Gain (to make bird call less loud)
- Background noise - very and less loud. We have taken some background from [here](https://www.kaggle.com/mmoreaux/environmental-sound-classification-50) and some 5 second clips directly from example audio. Finally we created such [background dataset](https://www.kaggle.com/vladimirsydor/cornelli-background-noises)
- LowFrequancy CutOff - we found out, that example audio has no lower frequency

Also we used MixUp - we add audios and take max from two one-hot targets. As it was done by  @ddanevskyi in [Freesound](https://www.kaggle.com/c/freesound-audio-tagging-2019/discussion/97926)

# Choose Final Blend

First we tried simply to Blend all our good models - 14 experiments (70 models) and it gave us 0.623 Public score and 0.669 Private score. But then we have taken 4 best experiments with external data and 3 best without external data, which gave us 0.624 Public and 0.67 Private scores

All Training details and inference of best Blend you can find in our [inference notebook](https://www.kaggle.com/vladimirsydor/4-th-place-solution-inference-and-training-tips?scriptVersionId=42796948) 

# Framework 

For training and experiment monitoring was used [Pytorch](https://pytorch.org/) and [Catalyst](https://catalyst-team.github.io/catalyst/) frameworks. Great thanks to @scitator and catalyst team! 

# P.S

Great thanks to my teammate - @khapilins
Also thanks to all DS community, especially to @ddanevskyi, @yaroshevskiy and @frednavruzov. They have taught me a lot and give inspiration to take part in Kaggle competitions.
Also thanks to all Kaggle team and community.
And happy Kaggling!
