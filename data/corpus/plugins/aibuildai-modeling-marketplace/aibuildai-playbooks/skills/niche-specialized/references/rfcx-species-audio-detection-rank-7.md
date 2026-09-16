# 7th place solution - Beluga & Peter

Competition: rfcx-species-audio-detection
Rank: #7
Source: https://www.kaggle.com/c/rfcx-species-audio-detection/discussion/220443

Thanks, Kaggle and RFCx, for this audio competition, and special thanks to my teammate @gaborfodor Without him, I probably would have given up a long time ago, somewhere at 0.8xx.

## Data preparation
Resampling everything to 32kHz and split the audio files into 3 seconds duration chunks. We used a sliding window with a 1-second step.

## Collecting more label
The key to our result was that Beluga collected tons of training samples manually. He created an awesome annotation application; you can find the details [here](https://www.kaggle.com/c/rfcx-species-audio-detection/discussion/220305). *Source code included*.


After the first batch of manually labeled examples, we quickly achieved 0.93x with an ensemble of a varying number of PANN (cnn14) models. 

## Input
We used mel-spectrograms as inputs with various `n_bin` (128, 192, 256, 288). Beluga trained PANN-cnn14 models with one input channel. For the other backbones (effnets, resnets, etc) I used three input channels with a simple trick:

- I used different `n_mel` and `n_fft` settings for every channel. E.g. n_mel=(128, 192, 256), n_fft=(1024, 1594, 2048). This results in different height images, so resizing to the same value is necessary.

We both used `torchlibrosa` to generate the mel-spectrograms.

## Augmentation
We used three simple augmentations with different probability:

##### Roll
```
np.roll(y, shift=np.random.randint(0, len(y)))
```

##### Audio mixup
```
w = np.random.uniform(0.3, 0.7)
mixed = (audio_chunk + rnd_audio_chunk * w) / (1 + w)
label = (label + rnd_label).clip(0, 1)
```

##### Spec augment
```
SpecAugmentation(time_drop_width=16, time_stripes_num=2, freq_drop_width=16, freq_stripes_num=2)
```

## Architectures
- PANN - cnn14
- EfficientNet B0, B1, B2
- Densenet 121
- Resnet-50
- Resnest-50
- Mobilnet v3 large 100

We trained many versions of these models with different augmentation settings and training data. Beluga used a PANN - cnn14 model (I think it is the same as the original) from his [Cornell solution](https://www.kaggle.com/c/birdsong-recognition/discussion/183407).

I trained a very similar architecture with different backbones and I used attention head from SED:

```
x = ...generate mel-specotrogram...
x = self.backbone.forward_features(x)

x = torch.mean(x, dim=2)

x1 = F.max_pool1d(x, kernel_size=3, stride=1, padding=1)
x2 = F.avg_pool1d(x, kernel_size=3, stride=1, padding=1)
x = x1 + x2

x = F.dropout(x, p=0.5, training=self.training)
x = x.transpose(1, 2)
x = F.relu_(self.fc1(x))
x = x.transpose(1, 2)
x = F.dropout(x, p=0.5, training=self.training)

(clipwise_output, norm_att, segmentwise_output) = self.att_block(x)

segmentwise_output = segmentwise_output.transpose(1, 2)
framewise_output = interpolate(segmentwise_output, self.interpolate_ratio)

output_dict = {
    "framewise_output": framewise_output,
    "clipwise_output": clipwise_output,
}
```

## Training
Nothing special. Our training method was the same for all of the models:
- 4 folds
- 10 epochs (15 with a higher probability of mixup)
- Adam (1e-3; *0.9 after 5 epochs)
- BCE loss (PANN version)
- We used the weights of the best validation LWRAP epoch for inference.

## Pseudo labeling
After we had an excellent ensembled score on the public LB (0.950), we started to add pseudo labels to our dataset. The result after we re-trained everything was 0.96x.


## Final Ensemble
Our final ensemble had 80+ models (all of them trained with 4-folds)
