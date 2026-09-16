# 23rd place solution

Competition: g2net-gravitational-wave-detection
Rank: #23
Source: https://www.kaggle.com/c/g2net-gravitational-wave-detection/discussion/275728

At first, I would like to thank my team members @vladimirsydor, @zekamrozek, @yakuben and @uulott for their work and organizers and the Kaggle community for the amazing experience we had during this competition.

##TL;DR
 Blend of 6 2D CNN + 3 1D CNN: 1x2D EfficentNet_b0 + 3 x _b3 + 2 x _b5 + 2 x 1D ResNet with transformer + 1D basic CNN

## Details
**Bandpass** was crucial for this data, we experimented with it a lot and tried different setups, those parameters showed better results then others we tried:
                `fmin=20`,
                `fmax=1024` or `fmax=500` or `fmax=600` depending on the model,
                `hop_length=4`,
                `bins_per_octave=12`,
                `filter_scale=0.5`,
                `pad=10`.

For images, we used only CQT and didn't experiment with CWT.

**Augmentations** that worked for us:
1. Wave augs:
- Random shift channel +-30;
- Hard MixUp of waves: `new_wave = (wave1 + wave2) / 2`  and `new_target = max(target1, target2)`;
- Adding Gaussian Noise with small std.


**Models**
For most of the competition, we were training EfficentNets, until around 1-2 last weeks we started experimenting with 1D models.

We found out, that increasing the input image would increase the performance even for EfficentNet_B0, so we experimented with it more. For us best result was when we used `stride=(1,2)` instead of default `(2,2)` with an unchanged shape of input. That allowed model to "stretch" the input itself across the frequency axis and give an effect of feeding higher resolution images to the network.

Our 2D models suffered from spoiled gradients in BatchNorm layers on first epochs, so to overcome that we used 2 procedures.  The first idea that worked was to substitute BatchNorm for InstanceNorm throughout the whole model.  2nd  idea that worked was that we would freeze all encoder layers but BatchNorm, and train only them plus Linear head for the first 2 epochs.

As was mentioned, we started working late on 1D models, mainly simple stacking Conv layers, but after merging with @uulott, he brought a really promising 1D ResNet model that performed on similar level as our best 2D ones (~0.879 LB).

Finally, all of our models benefited when we started finetuning them on a small learning rate for multiple epochs with augmentations turned off. That boosted us past 0.88 LB and was keeping us in silver as we kept going up LB the more models we fine-tune and blend with.

**Ensemble**
For ensembling, we used straightforward hyperopt blending. Adding Any 1D model boosted all of our blends.


**The final blend** consisted of 6 2D-CNNs and 3 1D-CNNs:

1.  EfficentNet-B5, successful augs, freeze 2 epochs, normalize after CQT; 
2. Same B5, but fine-tuned;
3.  fine-tuned EfficentNet-B3, all successful augs, freeze 2 epochs, normalize after CQT;
4. fine-tuned EfficentNet-B3, initial stride=(1,2)  GaussNoise aug, freeze 2 epochs, normalize after CQT;
5. fine-tuned EfficentNet-B3, initial stride=(1,2)  GaussNoise aug, freeze 2 epochs, normalize after CQT;
6. fine-tuned EfficentNet-B0 , InstanceNorm instead of BatchNorm, normalize after CQT;

7.  1D ResNet + Transformer head,  filter order 8 and CosineAnnealing;
8. 1D ResNet + Transformer head,  filter order 5 and ReduceLROnPlateau;
9. fine-tuned 6-layer CNN1d with MLP head, MixUp and GaussNoise, linear warm-up and cosine decay.

**P.S**. Other things we tried, but it didn't make it to the final blend:
- drop confident 1s before fine-tuning;
- InceptionTime for 1D + Transformer head;
- 8 layer custom 1D CNN;
- rexnet_100;
- EfficeintNet-B2 and other;
- resize CQT image into (512,512) or (256, 256).

Augmentations that we stopped using :
- Channels shuffle for an image in 2D models;
- Horizontal and vertical flip of images;
- normal MixUp.
