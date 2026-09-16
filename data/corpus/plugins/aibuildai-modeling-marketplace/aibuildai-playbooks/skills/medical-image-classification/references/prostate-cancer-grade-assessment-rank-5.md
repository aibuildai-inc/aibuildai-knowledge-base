# 5th place approach

Competition: prostate-cancer-grade-assessment
Rank: #5
Source: https://www.kaggle.com/c/prostate-cancer-grade-assessment/discussion/169213

I've learned a bunch from this contest - big thanks to the organizers and all who took a shot at solving the problem, and congrats to the winners! 
My approach is nothing novel but here goes.

### Models
My solution was an ensemble of semi-supervised ImageNet models based on @Iafoss' [concat tile pooling ](https://www.kaggle.com/iafoss/panda-concat-tile-pooling-starter-0-79-lb)

-  resnext50_32x4d_ssl: input 192x192, 256x256
-  resnext50_32x4d_swsl: input size 384x384

The only thing I changed was removing the final dropout layer and training the head for a few epochs before unfreezing the model.
And of course 
@haqishen's genius [BCE loss](https://www.kaggle.com/haqishen/train-efficientnet-b0-w-36-tiles-256-lb0-87).

### Data
I generated tile sizes 256 and 384 from the medium resolution based on @akensert's [optimized tiling](https://www.kaggle.com/akensert/panda-optimized-tiling-tf-data-dataset).

There was a performance trade-off between selecting more tiles and larger batch size so I settled on randomly sampling *k* tiles from the top *N* tiles for each epoch.

| model | input size | k | N | bs
| ----- | --- | -- | -- | --
| resnext50_32x4d_ssl  | 192 x 192 | 28 | 40 | 10
| resnext50_32x4d_ssl  | 256 x 256 | 32 | 40 | 6
| resnext50_32x4d_swsl  | 384 x 384 | 14 | 24| 6

Training with a smaller size(128) seemed to overfit while the larger size(512) was unstable because I had to lower the batch size

### Augmentations
Hue/saturation augmentations didn't improve CV so I stuck to affine transforms - rotations, flips, zoom, warp - all from the default fastai transforms. 
Randomly shuffling the tiles every other epoch also seemed to help.
