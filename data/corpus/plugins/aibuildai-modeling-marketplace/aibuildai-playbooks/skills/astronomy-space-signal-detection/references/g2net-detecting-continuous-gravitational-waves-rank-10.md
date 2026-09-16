# 10th place solution: CNN with pseudo labels

Competition: g2net-detecting-continuous-gravitational-waves
Rank: #10
Source: https://www.kaggle.com/c/g2net-detecting-continuous-gravitational-waves/discussion/376052

Thank you to the hosts and the organizers for having such a challenging competition!

My solution is based on a CNN. I incorporated various elements, but the following three items were particularly important.
- **normalization using RobustScaler**
- **multitask learning of target and frequency**
- **pseudo labels**

# Data

I used PyFstat to generate data with the same range of timestamps, frequency, and amplitudes as the test data. The signal data was generated with a signal depth (sqrtSX/h0) in the range of [1, 50]. The number of data for each is as follows.

| data type           | number of data |
| ------------------- | -------------- |
| gap noise           | 3000           |
| nonstationary noise | 2983           |
| signal              | 5400           |



# Preprocess

I created an image with shape (C,H,W) = (2, 360, 127) by performing the following processing on each of H1 and L1.

- Normalize the square of the amplitudes of H1 and L1 using `sklearn.preprocessing.RobustScaler`.
  - Since the real noise in the test data has large outliers, I use `sklearn.preprocessing.RobustScaler`, which is resistant to outliers.
- Align H1 and L1 timestamps in the same way as [G2NET large kernel inference](https://www.kaggle.com/code/laeyoung/g2net-large-kernel-inference), resulting in an image with shape (C,H,W) = (2, 360, 5760).
- Take a moving average to make the image size (2, 360, 5760) -> (2, 360, 127). 
- Clip the values of the channel with the larger maximum value.
  - Because the test set contains data with too large a difference in maximum values between H1 and L1.
- Finally, I standardize the image with entire data mean and standard deviation.


# Model

MultiOutput model predicting target and frequency. 

- Target loss is `nn.BCEWithLogitsLoss`
- Frequency//50 loss is `nn.CrossEntropyLoss` of 11 classes
- Architecture is tf_efficientnet_b5_ap
- Change the stride of the first conv layer of the model to (1,2) to scale up the image resolution

```
class CustomModel(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        self.net = timm.create_model("tf_efficientnet_b5_ap", 
                                     pretrained=pretrained, 
                                     num_classes=0, 
                                     in_chans=2)
        
        # decrease first conv's stride
        modules_iter = iter(self.net.modules())
        for module in modules_iter:
            if isinstance(module, torch.nn.Conv2d) and tuple(module.stride) == (2, 2):
                break
        module.stride = (1, 2)
        
        # Target 
        self.head1 = nn.Sequential(
            nn.Linear(self.net.num_features, 1)
        )
        
        # Frequency//50: 40-500Hz // 50
        freq_div_n = 50
        self.head2 = nn.Sequential(
            nn.Linear(self.net.num_features, (500 // freq_div_n) - (40 // freq_div_n) + 1)
        )

    def forward(self, x, labels=None):
        feat = self.net(x)
        y1 = self.head1(feat)
        y2 = self.head2(feat)
        return y1[:,0], y2
```

# Training
- cv: target StratifiedKFold(n_splits=5)
- optimizer: AdamW
- scheduler: warmup 0-3epoch(lr=4e-6->4e-4) + Cosine Annealing 3-100epoch(lr=4e-4->4e-6)


# Augmentation

- Mixup
  - mixup where data with target=1 is not mixed together
  - mixup with the overall mean value of the test set
- Horizontal and vertical lines of outliers that mimic the real data of the test set
- [Augmentation of 8th place in the last competition](https://www.kaggle.com/competitions/g2net-gravitational-wave-detection/discussion/275335)（LIGO Swap, Swap with Other Negative）
- GaussNoise
- Vertical Shift
- Torchaudio Masking（time masking, frequency masking）
- Horizontal Flip, Vertical Flip


# Pseudo Labels

Pseudo labels using only the simulation data from the test set and pseudo labels excluding real data with amplitudes that take on outliers were effective. 
It was better to train with soft labels than with hard labels.


# TTA

Horizontal Flip, Vertical Flip

 
# Ensemble

[Rank averaging](https://www.kaggle.com/c/ranzcr-clip-catheter-line-classification/discussion/205564) of submit.csv files from different models.
However, according to @poteman, Stacking seems to perform slightly better in private lb. (I heard about this after competition. poteman's submit is not used for final submission.)

# P.S.
**I did all the above solutions by myself. I did not get help from my teammates.**
Code: https://github.com/riron1206/kaggle-G2Net-Detecting-Continuous-Gravitational-Waves-10th-Place-Solution
