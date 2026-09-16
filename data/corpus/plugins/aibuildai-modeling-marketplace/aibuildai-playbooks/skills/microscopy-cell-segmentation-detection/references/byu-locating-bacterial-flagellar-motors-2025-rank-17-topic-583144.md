# 17th Place Solution - Ultralytics + Timm

Competition: byu-locating-bacterial-flagellar-motors-2025
Rank: #17
Source: https://www.kaggle.com/c/byu-locating-bacterial-flagellar-motors-2025/discussion/583144

# BYU Bacterial Flagellar Motors Competition - 17th Place Solution

First, we thank the competition hosts @andrewjdarleym, @braxtonowens and Kaggle staff for organizing this competition. Below, we introduce the solution of team **Sergio Alvarez + Paradox** -- @sersasj, @iamparadox.

## Context

- **Business context**: [Competition overview](https://www.kaggle.com/competitions/byu-locating-bacterial-flagellar-motors-2025/overview)
- **Data context**: [Competition data](https://www.kaggle.com/competitions/byu-locating-bacterial-flagellar-motors-2025/data)

## Background & Evolution

After achieving a good position in the CZII competition, our first approach (although not yet teamed up) was to try U-nets. However, this didn't work well, with the best model scoring only ~0.5 on the public leaderboard.

Next, I experimented with [Jun Koda's models](https://www.kaggle.com/code/junkoda/speed-up-inference), which used Timm backbones with simple classification and segmentation heads. I managed to achieve 0.770 on the public LB but got stuck.

After seeing discussions and public notebooks with good scores using YOLO, I decided to test it and achieved 0.792 on the public LB with YOLO10X. This got me thinking: if simple techniques with Timm backbones + a basic head got me 0.770, why not use the YOLO ultralytics framework that already has established augmentations, loss functions, SOTA neck for feature combination, and optimized bbox prediction heads? 

Paradox and I explored this approach, and here's our solution!

## Solution Overview

Our **17th place solution** consisted of 3 YOLO-like (if I can call them that) models using `convnextv2_base.fcmae_ft_in22k_in1k` as the backbone. We extracted features from P4, P3, and P2 for the neck and made predictions with a single head at P3 (stride / 8).

### Model Architectures

I started implementing Timm integration, but someone more intelligent than me had already done it! We just want to thank [yjwong1999](https://github.com/yjwong1999), who created the [PR that we based our approach on](https://github.com/ultralytics/ultralytics/pull/19609). We just added some stuff to it and created cfgs.

We developed two main model configurations that vary only in the "neck" design:

#### Model 1: 
- **Backbone**: `convnextv2_base.fcmae_ft_in22k_in1k`
- **Neck**: SPPF + C2PSA → upsample P4 and concatenate with P3 → head



#### Model 2:  
- **Backbone**: `convnextv2_base.fcmae_ft_in22k_in1k`
- **Neck**: SPPF + C2PSA → upsample P4, adaptive_avg_pool2d at P2, concatenate P3 and upsampled P4 → head




Paradox had the amazing idea of removing P5 scale and features after seeing [@tatamikenn's excellent notebook](https://www.kaggle.com/code/tatamikenn/reverse-engineering-yolo) about YOLO features. This modification improved our scores by ~0.015 on the public LB.
The adaptive_avg_pool2d/AVG block in the architecture is from [@yyyy0201 remarkable mhaf-yolo](https://github.com/yang-0201/MHAF-YOLO).

### Training Configuration

We trained with 80% of the images that contained motors with trust = 4. And additionaly used 80% [@bartley external dataset](https://www.kaggle.com/competitions/byu-locating-bacterial-flagellar-motors-2025/discussion/569921) dataset with trust = 0. The rest was used as validation.

We used these standard training parameters:

```python
results = model.train(
    data=str(yaml_path),
    epochs=40,
    batch=2,
    imgsz=960,
    optimizer='AdamW',
    lr0=1e-4,
    lrf=0.1,
    warmup_epochs=0,
    dropout=0.1,
    project=str(RUNS_DIR),
    exist_ok=True,
    name=f"fold{fold_idx}",
    patience=100,   
    save_period=1,
    val=True,   
    mosaic=0.5,
    close_mosaic=0,
    mixup=0.4,
    flipud=0.5,
    scale=0.25,
    degrees=45,
    seed=42,
    deterministic=True,
    label_smoothing=0.1,  # Note: isn't doing anything actually
    augment=True,
    device=0,
)
```

### Validation Strategy Challenges

Validation was problematic throughout the competition. We struggled to find correlation between our validation scores and public leaderboard performance, as our models were scoring 0.97+ in CV but showing different performance on the LB.

We also faced significant challenges in epoch selection, which led to [this discussion](https://www.kaggle.com/competitions/byu-locating-bacterial-flagellar-motors-2025/discussion/576756).

To mitigate the epoch selection problem, we used the model soup technique to average weights of multiple epochs.

Ultimately, we compared models using mAP50, mAP50-95, precision, and recall curves to estimate performance. This didn't work so well.

## Final Submission Details

### Ensemble Strategy

Our final ensemble consisted of three main components:

1. First model configuration with standard training
2. First model configuration with more augmentations:
   ```python
   T = [
       A.Blur(p=0.1),
       A.MedianBlur(p=0.1),
       A.ToGray(p=0.1),
       A.CLAHE(p=0.1),
       A.RandomBrightnessContrast(p=0.1),
       A.RandomGamma(p=0.1),
       A.ImageCompression(quality_upper=100, quality_lower=60, p=0.1),
       A.ShiftScaleRotate(p=0.1),
       A.GaussNoise(p=0.1),
       A.GaussianBlur(p=0.1),  
       A.UnsharpMask(p=0.1)
   ]
   ```
3. Second model configuration with standard training

### Inference Strategy

We used `concentration = 0.5` for selecting slices, meaning we predicted on half of the available slices to increase speed.

For aggregating predictions, we experimented with HDBSCAN clustering with the following parameters:
- min_samples: 1
- Cluster size: 4
- EPS: 50  
- YOLO threshold: 0.4

We also added a confidence adjustment based on cluster size to favor bigger cluster.


## What Didn't Work

- **Alternative Backbones**: We tested `tf_efficientnetv2_l.in21k_ft_in1k`, `resnext50_32x4d.fb_swsl_ig1b_ft_in1k`, and `caformer_b36.sail_in22k`. They achieved ~0.8 performance, none surpassed ConvNeXt base.
- **Larger Models**: ConvNeXt base was already quite huge. ConvNeXt large actually performed worse, so we stopped exploring larger architectures.
- **Different Head**: I've tried using ConvNeXt blocks in head, improved map50-95 but not in LB.


## Bonus PB

This probably happened to many teams, but we didn't select our top-scoring notebook for the private leaderboard. In fact, we didn't even select the top 5.

**Our True Best**: Using the second model configuration alone (a different epoch soup combination than what we used in the ensemble) achieved **0.856** on the private LB, which would have been in the gold zone, indicating that P2 features were actually really important.

[PB]


## Resources & Acknowledgments

- [@tatamikenn's YOLO features notebook](https://www.kaggle.com/code/tatamikenn/reverse-engineering-yolo)
- [Jun Koda's speed optimization](https://www.kaggle.com/code/junkoda/speed-up-inference) 
- [Model Soup paper](https://arxiv.org/abs/2203.05482)
- [MHAF-YOLO](https://github.com/yang-0201/MHAF-YOLO)
- [Bartley External Dataset](https://www.kaggle.com/competitions/byu-locating-bacterial-flagellar-motors-2025/discussion/569921)
- [Timm+Ultralytcs](https://github.com/ultralytics/ultralytics/pull/19609)
