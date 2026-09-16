# 37th place brief writeup

Competition: pku-autonomous-driving
Rank: #37
Source: https://www.kaggle.com/c/pku-autonomous-driving/discussion/127546

Thanks to the host and useful kernels and discussions. 
I learned a lot during this competition.

My code is based on [Center-Resnet Starter](https://www.kaggle.com/phoenix9032/center-resnet-starter).
My public LB was 0.094 and private LB was 0.086.

## Model
- model architecture is the same as Center-Resnet Starter Kernel except below
- feed mask image and x,y position into model
- predict heatmap with focal loss following CenterNet paper
- change regression target from (x,y,z) to (u-diff, v-diff, z) following CenterNet paper
- regress log(z) instead of z ∵ depth affects by multiplication and log(z) distribution is more balanced than z distribution

## Data Augumentation
- (x/z, y/z) position jittering
- slight gauss noise
- slight randomContrastBrightness

## Preprocessing / Post Processing
- masking prediction using given masks
- restore color distorted test images.
  for each image and each channel, stretch [0, '95 percentile value'] to [0, 255]
  - probably no effect on LB, pointed out by [this discussion](https://www.kaggle.com/c/pku-autonomous-driving/discussion/127060)
  
## Others
- replacing confidence by Y-position has no effect. At this point, I doubted the evaluation metric.
- remove corruputed 5 train images
- adaptive heatmap threshold to predict at least one car per an image. discarded it since LB does not change
- increase epochs and change scheduling to ReduceLROnPlateau
- 2x weights to regression targets to balance two types of losses. It improved LB
- add (x,y) position info as head input and add two 1x1 convs to head. 
   better localCV and private LB (my final sub score + 0.002)
   I discarded it since public LB was bad (my final sub score - 0.007)

## What did not work for me
- smaller input size (w,h = 1536,512)
- larger input size + grad accumulation (accumlation_step=2)
- deformable convolution V2 (maybe because of my poor modeling skill)
- bins with in-bins regression for pitch (bins=4) following CenterNet paper
- predict pitch from camera view following CenterNet paper

## What I should have tried
- improve predictions of large cars. Below may be relevant:
   9th solution : difference models for cars at difference positions
   5th solution : FPN network
- ensemble
- other backbones (DLA34 or resnet34 or efficientnet-b0)
- change the each-car distance threshold
- flip augmentation
- use pretrained model (and use mask info for loss calculation)

-------------------------
code is [here](https://github.com/lisosia/kaggle-pku-autonomous-driving)
