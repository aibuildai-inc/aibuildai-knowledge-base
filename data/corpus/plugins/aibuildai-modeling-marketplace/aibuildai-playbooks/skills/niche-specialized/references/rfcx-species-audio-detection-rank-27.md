# 27th place simple solution (0.932 public, 0.940 private) - dl_pipeline

Competition: rfcx-species-audio-detection
Rank: #27
Source: https://www.kaggle.com/c/rfcx-species-audio-detection/discussion/220306

**In summary:**
- Best single model (0.925 public lb): densenet121 features + fastai head
- Loss function: cross entropy
- Sampling: 128x1024 crops around true positive samples
- Spectrogram parameters: n mels 128, hop length 640, sample rate 32000
- Augmentations: clipping distortion, pitch shift and mixup
- Inference: Predict on crops with a small width (e.g. 128x128 instead of 128x1024 used for training) and calculate the max probability for each of the 24 classes.

**Introductory monologue**
First and foremost, this was an interesting competition and a good learning opportunity as it is often the case in Kaggle! One “problem” of this competition is that the test data was labeled with a different method and no samples of labeled test data were provided. This makes it difficult to get a sense of validation score and increases the danger of overfiting the public test results. In fact, I almost gave up on this competition when I realized this was the case. But eventually I decided to get back to it and work on a simple solution and on a python library – **dl_pipeline** ( https://github.com/mnpinto/dl_pipeline) – that I will use as a general framework for future kaggle competitions in general. Initially, the idea for dl_pipeline was just to keep my code more organized and more reusable but I figured that maybe there’s also some value in sharing it. 

**Data preprocessing**
Save all wave files in npy files with sample rate of 32000 Hz to save time.
```python
def audio2npy(file, path_save:Path, sample_rate=32_000):
    path_save.mkdir(exist_ok=True, parents=True)
    wave, _ = librosa.load(file, sr=sample_rate)
    np.save(path_save/f'{file.stem}.npy', wave)
```
I didn't convert the audio to spectrograms right away since I still want the ability to use audio augmentation on the waveforms.

**Augmentations and Spectrograms**
- First I create crops on the waveform including the true positive labels with a number of samples calculated so that the spectrogram will have a width of 1024. 

**Note:** Cropping before applying the augmentations is much faster than the other way around.

- Then for the waveform augmentations I used the **audiomentations** library (https://github.com/iver56/audiomentations). I ended up using just the following augmentations as based on public lb I didn't find that others were helping, although this would require a proper validation to take any conclusions. 

```
def audio_augment(sample_rate, p=0.25):
    return Pipeline([
        ClippingDistortion(sample_rate, max_percentile_threshold=10, p=p),
        PitchShift(sample_rate, min_semitones=-8, max_semitones=8, p=p),
    ])
```

**Note:** Some augmentations are much slower, for example, pitch shift and time stretch. When using those augmentations the probability of use makes a big difference in how long the training takes. 

- Then I searched the fastest way to convert the audio to spectrograms in the GPU and I ended up using **nnAudio** (https://github.com/KinWaiCheuk/nnAudio). Again, converting to spectrogram after the waveform is cropped is a nice gain in processing time.

**Model**
I tried several models but the one that got me a better result in the public leaderboard was densenet121 and the second-best ResNeSt50. One particularity is that I use for all the models the fastai head with strong dropout.

The fastai head (using `create_head(num_features*2, num_classes, ps=0.8)`).

```
(1): Sequential(
      (0): AdaptiveConcatPool2d(
        (ap): AdaptiveAvgPool2d(output_size=1)
        (mp): AdaptiveMaxPool2d(output_size=1)
      )
      (1): Flatten(full=False)
      (2): BatchNorm1d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (3): Dropout(p=0.4, inplace=False)
      (4): Linear(in_features=2048, out_features=512, bias=False)
      (5): ReLU(inplace=True)
      (6): BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (7): Dropout(p=0.8, inplace=False)
      (8): Linear(in_features=512, out_features=24, bias=False)
    )
```

**Training**
I guess code speaks more than words, particularly for those familiar with fastai:

```
bs = 32
learn = Learner(dls, model, loss_func=cross_entropy, metrics=[accuracy, lrap], cbs=cbs)
learn.to_fp16(clip=0.5);
learn.fit_one_cycle(30, 1e-3, wd=3e-2, div_final=10, div=10)
```

So in English, this is a one cycle learning rate schedule with **30 epochs** starting with lr=1e-4, increasing to 1e-3 and then decreasing back to 1e-4 following cosine anealing schedule. The **loss function** is the good old cross-entropy. Also, a **weight decay** of 3e-2 was used, a **gradient clip** of 0.5 and the train was done with **mixed-precision** so that my GTX 1080 can handle a **batch size** of 32 with 128x1024 image size.

One training epoch takes about 1 minute on my GTX 1080,  I guess it's not bad considering that I'm doing waveform augmentations on CPU that even with p=0.25 take some time.

**Inference**
This is the fun part because it was almost by mistake that I realised that making inference with smaller tiles is way better. I presume that this is the case because I'm training with cross-entropy for a single label problem but the test data is labelled with multiple labels. By using smaller crops the predictions are more multilabel friendly. The reason I've been using cross-entropy instead of binary cross-entropy and sigmoid for the typical multilabel problem is that for me the convergence was much faster using the cross-entropy approach and with better results. Maybe I made a mistake somewhere I don't know, I didn't investigate it in much detail.

- Run predictions on crops of the spectrogram with a width of 64, 128 and 256 (remember training was done with 1024), calculate the max probability for each class for each case (64, 128, 256) and the average of the 3 cases. The average of the 3 gave me public lb 0.928 on my best single model that I describe above, compared to 0.925 for just the 128 width inference.

- The final solution with public lb 0.932 and private lb 0.940 is an ensemble of a few training interations with some modifications. (I will update this tomorrow with more information).

**dl_pipeline**
And again the code for this solution is now public on this repo: https://github.com/mnpinto/dl_pipeline

The following code should correspond to the best single model solution but I need to check if I didn't mess up anything when cleaning the code:

```
#!/bin/bash
arch='densenet121'
model_name='model_0'
sample_rate=32000
n_mels=128
hop_length=640

for fold in 0 1 2 3 4
do
    echo "Training $model for fold $fold"
    kaggle_rainforest2021 --fold $fold --model_name $model_name \
        --model $arch --sample_rate $sample_rate --n_mels $n_mels \
        --hop_length $hop_length --bs 32 --head_ps 0.8 \
        --tile_width 1024 --mixup true >> log.train
done

for tw in 64 128 256
do
    echo "Generate predictions for $model with tile_width of $tw"
    kaggle_rainforest2021 --run_test true --model_name $model_name \
        --model $arch --sample_rate $sample_rate --n_mels $n_mels \
        --hop_length $hop_length --tile_width $tw \
        --save_preds true >> log.predict
done
```
----------------------------
Thanks for reading!
