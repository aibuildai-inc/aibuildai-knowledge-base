# 5th place solution

Competition: birdclef-2022
Rank: #5
Source: https://www.kaggle.com/c/birdclef-2022/discussion/327044

Many thanks to Kaggle and Cornell Lab of Ornithology for hosting such an interesting competition.

My solution is a reimplementation of the BirdCLEF 2021 [2nd place solution](https://www.kaggle.com/competitions/birdclef-2021/discussion/243463).
Special thanks to the new baseline team for publishing the amazing solution. And thanks to @julian3833 for sharing [the great baseline!](https://www.kaggle.com/code/julian3833/birdclef-21-2nd-place-model-submit-0-66)

# Model
My final submission is an ensemble of 9 models of different seed/fold/backbone, each model using last year's 2nd place solution method.

The backbones were:
* 4x `eca_nfnet_l0`
* 2x `tf_efficientnetv2_s_in21k`
* 2x `resnet34`
* 1x `convnext_tiny`

# Oversampling

To increase the number of files for minority classes(N<20), I split the training files by hand.
For example, `maupar` appears only in one file, but the file contains many different types of its songs and calls.
I cut it into segments of 10-30 seconds using a waveform editor ([Audacity](https://www.audacityteam.org/)) and created multiple training data from a single file.

The cut audio files were further augmented by applying effects such as noise reduction/reverb/gain/etc. to each segment in the waveform editor. Ultimately, I created 5-20 additional sample files per target minority class.


# Training

* Melspec
	1. window_size=1024, hop_size=320, fmin=50, fmax=14000, power=2, mel_bins=64, top_db=80 (nfnet, effnet, convnext)
	2. window_size=2048, hop_size=512, fmin=16, fmax=16386, power=2, mel_bins=256, top_db=80 (resnet)
* Data Augmentation(Waveform)
	* Background Noise (2020 nocall, 2021 nocall, freefield1010)
	* GaussianNoise
	* PinkNoise
	* NoiseInjection
	* RandomVolume
	* TimeShift
* Data Augmentation(Image)
	* SpecAug
	* CutOut
	* Lowpass
	* TranslateY (shift in freq dim, simulated pitch shift)
* Optimizer
	* AdamW, LR=1e-3, weight decay=1e-5, Cosine Anearling with warmup
* Loss
	* BCEWithLogitsLoss
	* [Weighted loss by rating](https://www.kaggle.com/code/julian3833/birdclef-21-2nd-place-model-train-0-66?scriptVersionId=88985374)
* 25-30 epochs
* Training target is the union of primary and secondary labels.
* All backbones were **pretrained on BirdCLEF2021** data.
* **micro-f1@0.1** for `scored_birds` was used for validation. Although a bit odd, the correlation between this score and Public/Private LB was not bad.


# Post Processing

* Probability averaged over previous and next chunks.
* Correct thresholds using [call/nocall binary classifier](https://github.com/ChristofHenkel/kaggle-birdclef2021-2nd-place/blob/main/configs/pp_binary_ext3_1.py) probabilities.
* Threshold optimization by percentile per clesses.

```python
# thresholds per class
threshold = pd.Series(np.percentile(test_df[SCORED_BIRDS].values, 90, axis=0), index=SCORED_BIRDS)
```

# What Did Not Work

* SED models - could not make it past 0.72 for me
* CoordConv
* MC Dropout
* Label Smoothing
* Undersampling for majority classes
* ViT / Swin backbone
