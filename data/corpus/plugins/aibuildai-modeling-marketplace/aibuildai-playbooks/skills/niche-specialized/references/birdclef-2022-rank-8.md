# 8th place solution. And gold on last attempt

Competition: birdclef-2022
Rank: #8
Source: https://www.kaggle.com/c/birdclef-2022/discussion/327019

It was my first Kaggle competition on which I spent a lot of time. I'm honor to be part of it, big thanks to hosts, Kaggle and all participants.

## First step

1. I started with @kaerunantoka public [notebook](https://www.kaggle.com/code/kaerunantoka/birdclef2022-use-2nd-label-f0) with change mel spectrogram hop length and input shape [224, 512] and add **class weights** to a submission.
Weights array is 500 divided to amount of each birds species, and clamp it to max value 10. Then I multiply model output to weights array. **0.77** Public LB.

2. I tried this approach with secondary labels with 0.3, 0.4 and 0.5 labels. And with ensembling I'v got **0.79** on LB

##### Augmentations
For waveform:
```
Compose([OneOf(
     [AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.011, p=1),
            NoiseInjection(p=1, max_noise_level=0.04)], p=0.4),                                
            PitchShift(min_semitones=-4, max_semitones=4, p=0.1),
            Shift(min_fraction=-0.5, max_fraction=0.5, p=0.1),
            Gain(min_gain_in_db=-12, max_gain_in_db=12, p=0.2),    
            Normalize(p=1.) ])
```
For spectrogram:
 ```
torchaudio.transforms.FrequencyMasking(24)
torchaudio.transforms.TimeMasking(96)
```
##### Model 
SED with tf_efficientnet_b0_ns backbone

##### Training
* Epochs = 30
* LR = 0.001
* weight_decay = 0.0001
* dropout = 0.4
* Loss: Focal BCE Loss
* Optimizer: Adam (betas=(0.95, 0.999))
* Scheduler: CosineAnnealingLR

##### Validation
I validate models on 7 folds CV with f1-macro metric. Thresholds for f1: [0.5, 0.1, 0.15, 0.20, 0.25, 0.30, 0.35, 0.4, 0.45, 0.5]. Best checkpoints chosen by 0.3 and 0.5 thresholds.

##### What didn't work on this step
1. Noise reduction
2. Oversampling rare birds
3. Trim long silence with noise reduction
4. Weighted loss
5. 3 channels input, where 1st channel is power_to_db, 2nd melspectrogram, 3rd normalized melspectrogram


## Second step
1. I manually trim all segments without bird's sounds from the scored audios, except skylar and houfin, on them I processed only 70 records per class. 
2. Splited data to 15 seconds chunks for audios with length less than 1 minute and 30 seconds chunks for more than 1 minute. Got ~40.000 records.
3. Used weights arrays with 0.25-0.75 power to reduce their impact to inference.
4. Training on this data with previous pipeline and ensemble with previous models gives me **0.81** on LB

##### Data preprocessing
Same

##### Augmentations
Same

##### Model 
SED with tf_efficientnet_b0_ns backbone

##### Training
* Epochs = 60
* LR = 0.0013
* weight_decay = 0.0001
* dropout = 0.4
* Loss: Focal BCE Loss
* Optimizer: Adam (betas=(0.95, 0.999))
* Scheduler: CosineAnnealingLR

##### What didn't work on this step
1. PaSST model
2. AST model
3. Linear head models
4. PaSST preprocessing

## Third step
1. I used similar with AST preprocessing.
2. Every epoch from all data randomly choose up to 300 records for every class.
3. Train on random 5 seconds crop, validate on first 5 seconds.
4. Secondary labels 0.4
5. Class weights array clamped with max value 8. And used with power 0.6. 
6. Used mixup 0.4 for first 15 epochs, mixup 0.07 for 16-22 epochs, and no mixup for others.
7. Ensembling of this approach with previous gives me **0.82** LB

##### Data preprocessing
```
waveform, sr = ta.load(filename)
waveform = crop_or_pad(waveform, sr=SR, mode=self.mode)
waveform = waveform - waveform.mean()
waveform = torch.tensor(self.wave_transforms(samples=waveform[0].numpy(), sample_rate=SR)).unsqueeze(0)
fbank = ta.compliance.kaldi.fbank(waveform, htk_compat=True,          sample_frequency=SR, use_energy=False, window_type='hanning',                 num_mel_bins=self.melbins, dither=0.0, frame_shift=9.7)
fbank = (fbank - self.norm_mean) / (self.norm_std * 2)
```
Mean and std calculated for all train dataset.

##### Model 
SED with tf_efficientnet_b0_ns backbone

##### Augmentations
Same

##### Training
* Epochs = 85
* mixup 0.4, when epochs < 15
* LR = 0.0008
* weight_decay = 0.0001
* dropout = 0.4
* Loss: Focal BCE Loss
* Optimizer: Adam (betas=(0.95, 0.999))
* Scheduler: CosineAnnealingLR

##### Postprocessing
* Mean-median averaging of predictions
```
          full_med = np.median(full_events, axis=1)                    
          full_mean = np.mean(full_events, axis=1)
          full_events = np.mean(np.array([full_med, full_mean]), axis=0) 
```
* Max adder.
```
       logits_max = full_events.max(0)
       for jk in range(full_events.shape[1]):
           if logits_max[jk] > threshold * 2.5:
               full_events[:, jk] += threshold * 0.5
```

## Story of gold medal
When there were 2 days left until the end of the competition, I thought about using a pseudo labeling. And separate my data by 0.8 threshold to remove all noisy records. Only ~15.000 left from ~40.000. Then I trained a new model on this data. Because there were not enough time I trained model on all train data and validate it on some random part and random part of only scored birds. 
All training and preprocessing parameters was the same with previous.

It was last day of competition and only 5 submissions. So the first 3 I lost because i forgot to make model.eval() :)
4th attempt I used with single new model and get **0.79** LB.
And the last one was mean between my best previous attempt and this pseudo labeling new model.

And it gave me **0.79** private, when my previous best private was **0.78**
