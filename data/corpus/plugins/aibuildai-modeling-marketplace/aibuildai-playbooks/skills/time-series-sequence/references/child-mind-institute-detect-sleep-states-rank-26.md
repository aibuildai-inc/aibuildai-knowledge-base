# 26th solution - 1D Unet with 1D CNN, Hybrid CNN-Transformer and WBF

Competition: child-mind-institute-detect-sleep-states
Rank: #26
Source: https://www.kaggle.com/c/child-mind-institute-detect-sleep-states/discussion/459671

Many thanks competition host and Kaggle for another interesting challenge. A big congrats to all the winners !

As usual, I have a good time learning during this competition. In this post, I will brieftly describe my solution.

All of my models learn to predict 3 targets:
+ Lower resolution heatmap of eventness (transition/critical points)
+ Offset to event (from coarser grid coordinates to pixel-level event step)
+ Binary segmentation mask for sleep period

 
Input is a window of 2 or 4 days, where first and last 4 hours (near border) is still included in the input sequence, but ignored in loss computation. The intuition is to prevent ambigous near border region since we need more context around to judge.
The longer crop window, the higher change of overfit, but better context to make accurate prediction. Especially for Hybrid CNN-Transformer with much larger model capacity, model quickly overfit when using large window. I consider crop_len is also an important hyperparam.

Segmentation mask prediction was used as an auxiliary target and was not ultilized in the inference pipeline.

[target visualization]

The offset prediction then be used as "error correction" to provide sub-pixel level accuracy despite large target strides (4, 8, 16, ..), inspired by Human Pose Estimation and Object Detection models. It looks nearly the same as CCRF ( Combined Classification and Regression Form) used in [UDP-Pose](https://github.com/HuangJunJie2017/UDP-Pose).



## Model
1D Unet with hierarchical 1D encoder. Two type of encoders: pure 1D-CNN or Hybrid CNN-Transformer. Output prediction is from 1/4 stride from original resolution (that is, make prediction per 3 steps or 15 seconds)


[architecture]


- Tried various stem type, Squeezeformer stem give slightly better/consistent CV

### 1D-CNN
Simply try to transform 2D-CNN image models to 1D version, replace all BatchNorm -> LayerNorm
- Crop_len = 4 days
- Basic Block: MobilenetV3, expand_ratio = 2, constant dim = 96 across stages
- Depth = 8 with stage computation ratios [3, 3, 3, 3, 3, 3, 3, 3] (VGG style ?) 
- ECA/SE attention
- backbone_dropout = 0.4, head_dropout = 0.4

### Hybrid CNN-Transformer
- Crop_len = 2 days
- SqueezeformerBlock with RoPE from top solutions in [previous ASL Fingerspelling competition](https://www.kaggle.com/competitions/asl-fingerspelling/discussion/434485)
- Two architecture included in final submission: hierarchical/non-hierarchical Transformer


## Loss
Simple weighted combination of:
- BCE for heatmap
- MSE or L1Loss for offset
- BCE for segmentation
 
Best weights: total_loss =  1.0 * heatmap + 0.2 * offset + 0.2 * segment


## Training
- FP16, Model EMA, batch size 16
- Max updates: 500 * 40 for CNN, 200 * 60 for Hybrid Encoder
- Optimizer: Lookahead AdamW + Cosine LR


## Post processing
- During training, track validation score outputed from simple NMS (Point-based NMS).
- Submission using Point-based version of **Weighted Boxes Fusion (WBF)** to ensemble multiple models (after NMS).

## Results

[local cv results]

# What worked
- Tuning sigma for Gaussian-like heatmap target: very important and give significant boost. Best sigma = 5 * 12 = 5 minutes;  Gaussian width = 2 * 3 * sigma = 30 minutes.
- Auxiliary segmentation loss: + ~0.01 AP
- Tuning loss weights
- Try various architecture designs: depth, dim, norm, activation, stage computation ratios,..
- Simple downsample the sequence length by reshaping: worked better than Convolution downsampling on raw sequence. Perhaps help avoid overfiting as mentioned in [this solution](https://www.kaggle.com/competitions/tlvmc-parkinsons-freezing-gait-prediction/discussion/416026)
- Model EMA: significantly stablize training and allow training for more epochs, crucial for Hybrid encoder
- Weighted Boxes Fusion: with 2 models ensemble, +1.3 OOF AP from 77.0 -> 78.3


# What not worked
- Feature engineering: tried various features (on single fold) but no boost. One make me suprised is just single feature 'anglez_sign' (+1 or -1) could archive CV=60.89
- Augmentation: just implement and tried some simple methods (Flip, Noise, Resampling, TimeStretch, ..) in the last day, also seem no boost and need more tuning.
- Decoupled head: Decoupled head for objectness & offset usually boost Object Detection model, but in this case simple single linear head with output_dim=5 work best.
- Ultilize segmentation prediction in post processing: no clear boost
