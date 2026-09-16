# 27th solution

Competition: child-mind-institute-detect-sleep-states
Rank: #27
Source: https://www.kaggle.com/c/child-mind-institute-detect-sleep-states/discussion/459761

I implemented this solution based on @tubotubo's source code. Thank you for sharing!

## Input
- anglez（normalized）
- enmo（normalized）
- hour_sin
- hour_cos
- minute_sin
- minute_cos

## Model
- Label
    1. awake
        - BCEWithLogitsLoss
    2. onset, wakeup
        - BCEWithLogitsLoss
    - loss = (awake loss) * (1 - 0.88) + (onset, wakeup loss) * 0.88

- Data Augmentation
    - GaussianNoise
        - min_amplitude: 0.01
        - max_amplitude: 0.1
    - [Zebra Mixup](https://www.kaggle.com/competitions/g2net-gravitational-wave-detection/discussion/275335)

- encoder: UNet
| feature extractor | decoder | downsample | CV |
| --- | --- | --- | --- |
| LSTM | UNet1D | 2 | 0.7611 |
| CNN | UNet1D | 2 | 0.7628 |
| Spectrogram | UNet1D | 2 | 0.7463 |
| CNN | Transformer | 6 | 0.7480 |
| Spectrogram | Transformer | 6 | 0.7725 |
| 2CNN+ time feature embedding | UNet1D | 2 | 0.7705 |

- 2CNN+time feature embedding
  - anglez and enmo are input separately into CNN feature extractors
  - In this model, month, hour, and minute are input as categorical variables.
  - month, hour and minute are converted into embeddings and concatenated with the encoder output before being input to the decoder.

```python
month = self.month_embedding(month) # (batch_size, 4, n_timesteps)
hour = self.hour_embedding(hour) # (batch_size, 8, n_timesteps)
minute = self.minute_embedding(minute) # (batch_size, 16, n_timesteps)
x = x[:, 3:, :] # (batch_size, n_channels, n_timesteps)
x1 = self.feature_extractor1(x[:, :x.shape[1]//2, :])  # (batch_size, n_kernel, height, n_timesteps)
x2 = self.feature_extractor2(x[:, x.shape[1]//2:, :])  # (batch_size, n_kernel, height, n_timesteps)
x = torch.cat([x1, x2], dim=2) # (batch_size, n_kernel, height*2, n_timesteps)

x = self.encoder(x).squeeze(1)  # (batch_size, height, n_timesteps)
x = torch.cat([x, month, hour, minute], dim=1) # (batch_size, height+28, n_timesteps)
logits = self.decoder(x)  # (batch_size, n_timesteps, n_classes)
```

## PostProcess
- scipy.signal find peaks (after ensemble)
    - Chose the smallest possible value for height (0.003)
    - distance optimized with optuna
- overlap inference

## Ensemble
- 4fold * 6model
- Ensemble with weighted output heatmaps
- Weights optimized with optuna
  - ensemble (mean) -> cv: 0.795
  - ensemble (weighted mean) -> cv: 0.7981 (public: 0.753 private: 0.809)
