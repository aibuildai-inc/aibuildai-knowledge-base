# 19th place solution

Competition: g2net-gravitational-wave-detection
Rank: #19
Source: https://www.kaggle.com/c/g2net-gravitational-wave-detection/discussion/275440

Thank you to the host and Kaggle for organizing the competition. Thank you to all my teammates( @sunakuzira @kzkt0713 @keiichimase @kanbehmw )and participants. I would like to share a summary of our solution.

# CQT
| Model | oof | Private LB |
| :---: | :---: | :---: |
| resnet34d | 0.8794 | 0.8797 |
| tf_efficientnet_b2_ap | 0.8802 | 0.8803 |
| tf_efficientnetvv2_b1 | 0.8800 | 0.8800 |
| tf_efficientnet_b2_ap | 0.8801 | 0.8800 |
| resnet34 | 0.8787 | 0.8786 |

#### preprocessing
- standardization: We computed statistics on standardization using entire data set.
- image resize: 276x513 or  207x513
- channel add: We added a channel with added and subtracted LIGO waves.
  ```
  waves = np.stack([
      waves[0], waves[1], waves[2], 
      waves[0]+waves[1], 
      waves[0]-waves[1]])
  ```

#### nnAudio.Spectrogram.CQT1992v2
```
CQT1992v2(sr=2048, fmin=20, fmax=1024, hop_length=8or4, window="flattop")
```

#### Augmentation
- augmentation by mixed waveforms
```
class CustomDataset(Dataset):
    def __init__(self, train, ...):
        self.train = train.reset_index(drop=True).copy()
        self.labels = train["target"].values
        self.train_target0 = train[train["target"] == 0]
    ...
    def __getitem__(self, index):
        y_true = self.labels[index]
        y_true = torch.tensor(y_true).float()
        ...
        if np.random.rand() > 0.5:
            if y_true == 0:
                sample = self.train.sample()
            else:
                sample = self.train_target0.sample()
            wave2_path = sample.iloc[0]['file_path']
            waves2 = self.load_img(wave2_path)

            waves = waves + waves2
            if sample.iloc[0]['target'] == 1:
                y_true = torch.tensor(1).float()
```
  
- albumentations
```
albumentations.ShiftScaleRotate(p=0.5, shift_limit=0.0, scale_limit=0.2or0.3, rotate_limit=0)
```


# 1dCNN
| Model | oof | Private LB |
| :---: | :---: | :---: |
| 1dCNN | 0.8753 | 0.8769 |

The architecture of 1dcnn used Public Kernel.( https://www.kaggle.com/kit716/grav-wave-detection/data?select=g2net_models.py )

#### preprocessing
- standardization: We computed statistics on standardization using entire data set.
- band pass: scipy.signal.butter(6, (35, 800), btype='bandpass', fs=2048)
- channel add: Same as CQT.

#### Augmentation
- random invert waves
```
if np.random.rand() > 0.5:
    waves = waves*-1
```
- albumentations
```
albumentations.ShiftScaleRotate(p=0.5, shift_limit=0.0, scale_limit=0.1, rotate_limit=0)
```

#### TTA
- We used TTA with the same Augmentation as train.

# pseudo label
- We used a high LB submission.csv as a **soft label**.

# weight optimize blending
- We used oof to optimize the weights. We used Public Kernel.( https://www.kaggle.com/itsuki9180/g2net-oof-weight-optimizer )
- **6model oof : CV:  0.8812, Private LB: 0.8805**
