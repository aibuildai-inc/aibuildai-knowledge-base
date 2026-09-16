# 9th place solution

Competition: czii-cryo-et-object-identification
Rank: #9
Source: https://www.kaggle.com/c/czii-cryo-et-object-identification/discussion/561431

First of all, I would like to express my sincere gratitude to the competition host and the Kaggle staff for organizing such a fascinating competition. I thoroughly enjoyed this competition and learned a great deal in the process!

Furthermore, I'd like to thank @hengck23 and @davidlist . @hengck23 's discussion and notebook were the starting point for my solution, and Many of the discussions and comments from @davidlist were extremely helpful.

## Summary

My solution is simple and straightforward. I performed segmentation using a 3D ConvNeXt-like model and further employed an ensemble of as many models as possible. Subsequently, I computed the centroids of the particles using cc3d and curated the predicted centroids with DBSCAN. 

## Segmentation Mask

I used ground truth masks with an adjusted radius for each particle. As a result, the public leaderboard score jumped 0.02~0.04. Concletely, I adjust mask size as shown in the below table; 

| apo-ferritin(r=60) | beta-galactosidase(r=90) | ribosome(r=150) | thyroglobulin(r=130) | virus-like-particle(r=135) |
| --- | --- | --- | --- | --- |
| 60/2 | 90/2 |  150/3 |  130/3 | 135/3 |

Since the competition metric requires that the predicted centroid falls within a radius of r×0.5 from the ground truth centroid, I believe that multiplying the radius of each particle by a factor of 0.5 or less is reasonable. 

## Model Architecture

In this section, I explain my model architecture both encoder and decoder. 

### Encoder

To implement the encoder, I started with the ConvNeXt, since ConvNeXt is very powerful and fast. Then, I customized the model to suit the task that is to predict small particles. These changes from the original ConvNeXt are below;

- 2D to 3D for all convolutional layer
- 4x4 stem -> 2x2 stem. Because the particles are very small, I believed that a large stem would adversely affect the model's predictions, especially apo-ferritin and beta-galactosidase. This modification slowed down the model's inference, but it improved its cross-validation performance. 
- 7x7 kernel size in conv block ->3x3. The reason why I modified like this is the same as the stem. This modification improved both inference speed and cv. 
- (3, 3, 9, 3) number of block -> (3, 3, 3, 3). this modification is to reduce a number of parameters and get more inference speed. cv did not decrease. 

### Decoder

The flow of the decoder is based on U-Net. The implementation of conv block is shown in the following code; 

```python
# conv block for decoder
class ConvBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, in_channels, kernel_size=3, padding=1, bias=False, groups=in_channels)
        self.norm1 = nn.GroupNorm(num_groups=1, num_channels=in_channels)
        self.act1 = nn.GELU()
        self.conv2 = nn.Conv3d(in_channels, in_channels*3, kernel_size=1, bias=False)
        self.conv3 = nn.Conv3d(in_channels*3, out_channels, kernel_size=1, bias=False)

    def forward(self, x):
        # x: (B, C, D, H, W)
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.conv2(x)
        x = self.act1(x)
        x = self.conv3(x)
        return x
```
And the implementation of the decoder is shown in the following code; 

```python
class Decoder3D(nn.Module):
    def __init__(self,
                 encoder_dims=[64, 128, 256, 512],
                 decoder_dims=[32, 64, 128, 256],
                 out_channels=5):
        super().__init__()
        self.up3 = nn.Upsample(scale_factor=(2,2,2), mode='trilinear', align_corners=True)
        #self.up3 = nn.ConvTranspose3d(encoder_dims[3], encoder_dims[3], kernel_size=2, stride=2)
        self.dec3 = ConvBlock3D(in_channels=encoder_dims[3] + encoder_dims[2], out_channels=decoder_dims[2])
       ...

    def forward(self, features):
        x, f0, f1, f2, f3 = features

        # --- 1) f3 -> f2 ---
        d3 = self.up3(f3)
        d3 = torch.cat([d3, f2], dim=1)
        d3 = self.dec3(d3)
        ...
```

Since the ground truth for segmentation is simply a sphere, I used a basic `nn.Upsample` for upsampling instead of `nn.ConvTranspose3d` to reduce parameters. 

## Other Details

- loss dunction is bce
- My inference code is based on @hengck 's [this notebook](https://www.kaggle.com/code/hengck23/1-hr-fast-2d-3d-unet-resnet34d-scanner-tta) and the idea of DBScan came from @linheshen 's [notebook](https://www.kaggle.com/code/hengck23/1-hr-fast-2d-3d-unet-resnet34d-scanner-tta). 
- The window size for inference is (32, 320, 320) (overwrap of z axis is 6). The window size for training is (32, 256, 256). 
- TTAs are rot90, 180 and 270. 
- preprocessing is only normalization
```python
    def normalize_numpy(self, x):
        lower, upper = np.percentile(x, (1, 99))
        x = np.clip(x, lower, upper)
        x = x - np.min(x)
        x = x / np.max(x)
        return x
```
- Augmentations for training are rot90, 180, 270, and xyz flip. 

## What did not work

- pretraining with synthetic data provided by host
- more and more smaller mask(e.g. factor=0.25 for large particles and factor=0.4 for small particles)
- flip tta instead of rot90

## Code & Model Checkpoint
inference code: https://www.kaggle.com/code/wadakoki/czii-9th-inference
training code: https://colab.research.google.com/drive/1OG5Rap9zUkKG6fnGZ8kP0JwwFZfSTgCc?usp=sharing
model checkpoint: https://www.kaggle.com/datasets/wadakoki/czii-9th-final-models/data
generated mask: https://www.kaggle.com/datasets/wadakoki/czii-mask-small
code for mask generation: https://www.kaggle.com/code/wadakoki/czii-gen-small-masks

**Requirements for training code on google colab**
- GPU: L4
- RAM: High Memory
