# 17th Place Solution : file-level post-process

Competition: birdsong-recognition
Rank: #17
Source: https://www.kaggle.com/c/birdsong-recognition/discussion/183255

https://github.com/xins-yao/Kaggle_Birdcall_17th_solution
## Feature Engineering
- Log MEL Spectrogram
```
sr = 32000
fmin = 20
fmax = sr // 2

n_channel = 128
n_fft = 2048
hop_length = 512
win_length = n_fft
```

## Augment
- Random Clip
 randomly cut 5s clip from audio and only keep clips with SNR higher than 1e-3
  ```
  def signal_noise_ratio(spec):
      spec = spec.copy()

      col_median = np.median(spec, axis=0, keepdims=True)
      row_median = np.median(spec, axis=1, keepdims=True)

      spec[spec < row_median * 1.25] = 0.0
      spec[spec < col_median * 1.15] = 0.0
      spec[spec > 0] = 1.0

      spec = cv2.medianBlur(spec, 3)
      spec = cv2.morphologyEx(spec, cv2.MORPH_CLOSE, np.ones((3, 3), np.float32))

      spec_sum = spec.sum()
      try:
          snr = spec_sum / (spec.shape[0] * spec.shape[1] * spec.shape[2])
      except:
          snr = spec_sum / (spec.shape[0] * spec.shape[1])

      return snr
  ```
- MixUp
 mixup over LogMelSpec with beta(8.0, 8.0) distribution
 beta(0.4, 0.4) and beta(1.0, 1.0) will raise more TruePositive but lead to much more FalsePositive
- Noise
 add up to 4 noises with independent probabilities and scales in waveform
 noises were extracted from training samples
  ```
  def signal_noise_split(audio):
    S, _ = spectrum._spectrogram(y=audio, power=1.0, n_fft=2048, hop_length=512, win_length=2048)
    
    col_median = np.median(S, axis=0, keepdims=True)
    row_median = np.median(S, axis=1, keepdims=True)
    S[S < row_median * 3] = 0.0
    S[S < col_median * 3] = 0.0
    S[S > 0] = 1
    
    S = binary_erosion(S, structure=np.ones((4, 4)))
    S = binary_dilation(S, structure=np.ones((4, 4)))
    
    indicator = S.any(axis=0)
    indicator = binary_dilation(indicator, structure=np.ones(4), iterations=2)
    
    mask = np.repeat(indicator, hop_length)
    mask = binary_dilation(mask, structure=np.ones(win_length - hop_length), origin=-(win_length - hop_length)//2)
    mask = mask[:len(audio)]
    signal = audio[mask]
    noise = audio[~mask]
    return signal, noise
  ```
## Model
- CNN
 9-layer CNN
 average pooling over frequency before max pooling over time within each ConvBlock2D
 SqueezeExcitationBlock within each ConvBlock2D
 pixel shuffle: (n_channel, n_freq, n_time) -> (n_channel * 2, n_freq / 2, n_time)
- CRNN
 2-layer bidirectional GRU after 9-layer CNN
- CNN + Transformer Encoder
 Encoder with 8-AttentionHead after 9-layer CNN

## Trainng
- Label Smooth: 0.05 alpha
- Balance Sampler: randomly select up to 150 samples of each bird
- Stratified 5Fold based on ebird_code
- Loss Function: BCEWithLogitsLoss
- Optimizer: Adam(lr=1e-3)
- Scheduler: CosineAnnealingLR(Tmax=10)

## Post-Process

if model gets a confident prediction of any bird, then lower threshold for this bird in the same audio file
- use thr_median as initial threshold
- use thr_high for confident prediction
- if any bird with probability higher than thr_high in any clip, lower threshold to thr_low for this specific bird in the same audio file
