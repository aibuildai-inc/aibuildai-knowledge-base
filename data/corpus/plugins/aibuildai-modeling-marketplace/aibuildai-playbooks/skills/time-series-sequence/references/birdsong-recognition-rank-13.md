# 13th place digest

Competition: birdsong-recognition
Rank: #13
Source: https://www.kaggle.com/c/birdsong-recognition/discussion/183436

It was my first audio competition and I really enjoyed it! Thanks to organizers and Kaggle for it.
Thanks to my teammates @tikutiku @zfturbo the great collaboration, it was a pleasure and I've learnt a lot again.

Here is a digest of our solution.

**Main Pipeline** (see image below):
- MEL-Spectrogram (torchaudio) with differents settings to capture most birds frequency shapes.
- Most processing/augmentation occurs in GPU to speed up training and inference.

**Time Augmentations**:
- Pink Noise/White noise
- Time roll
- Volume gain
- PitchShift
- Additional bandpass filters,lowcut (1kHz-2.5kHz), highcut (10kHz-15kHz)
- Background/Ambient noise Mixup
- Other birds Mixup (2 to 3 birds)

**Spec/Image augmentations**:
- Frequency/time masking
- Color jitter

**CNN backbones**:
- EfficientNet B1 and B2
- SEReseXt26
- ResneSt50
- Optional GRU layer to learn about time sequences (see [TALNet Sound Event Classifier](https://github.com/srvk/TALNet))
- Different pooling to apply SED concept with either Attention block or max pooling
We also had one model based on 1D signal only (DensetNet1D)

What did not work well (it worked but was disapointing):
- Wavegram + MEL-Spectrogram model (was bad with soundscape)
- Create large image with 3x2 grid spectrogram (was bad on inference)

**Data**:
- Train audio provided with duration outliers removed
- Additional xeno-canto data (from Vopani dataset, thanks @rohanrao for your clean dump)
- Distractors (NoBird/Nocall/Ambient) 10s slices (from freefield1010)
We've built our own validation data by mixing test_audio/birds/noise/nocall to try to correlate LB. It correlated a bit but was not enough to give trust in it. Too bad as it was key for this competition.

**Training procedure**:
- Stage1: Train a few models with 5s slices picked randomly, then save birds probabilities on CV OOF, ensemble all OOF models results to generate "*hot*" slices with high probabilities.
  It allowed to reach public LB=0.582
- Stage2: Train more models with only such 5s *hot slices*.
  It allowed to reach public LB=0.596
While reading other's solutions, we should have tried another stage with "very hot" slices to have a super clean train dataset.

**Final ensemble**:
Ensemble is a combination of models (8 to 10) with union strategy and with per-model threshold tuned on public LB (it overfitted for sure).
Idea of union was to capture most of the TP (better for metric used in this competition), drawback is that we capture FP too.
Inference was fast, we cached resampled audio in memory, and (almost) full GPU pipeline helped. It ran in around 1h30 so we still had room for more models.

**Post-processing**:
We have a "nobird" majority vote to try to remove FP (force nocall) as some models have been trained with 265 classes instead of 264 (class 265 = nobird)

Both our final submissions reached similar score on private LB.



Final words to conclude: Congratulations to top teams! and thanks to @hidehisaarai1213, @hengck23 for their sharings.
