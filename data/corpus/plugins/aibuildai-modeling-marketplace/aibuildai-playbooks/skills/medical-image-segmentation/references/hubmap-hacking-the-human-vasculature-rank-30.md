# 30th place solution

Competition: hubmap-hacking-the-human-vasculature
Rank: #30
Source: https://www.kaggle.com/c/hubmap-hacking-the-human-vasculature/discussion/428372

## Overview

@ayberkmir and I joined the competition very late. I wasted my first week because I didn't read the metric properly and submitting not accordingly. We didn't have time to try so many diverse things so it will be quite simple.

## Validation

Ayberk was using stratified single train/test split (stratified on blood vessel count) and I was using 4 fold leave-one-WSI-out cross-validation. Surprisingly both of them worked on private leaderboard.

## Models

I mainly worked on Mask R-CNN and Cascade Mask R-CNN with ResNeXt101 backbone and Ayberk worked on YOLOv7 and v8 models.

## Training

I dropped duplicate annotations by simply suppressing overlapping boxes with 0.99 IoU when I was creating datasets. I also included glomeruli annotations and used unsure annotations as blood vessels. I thought if I treat unsure as blood vessels, training would be more stable. Ayberk didn't use glomeruli or unsure annotations and YOLO was already handling duplicate annotations.

Since the WSIs were kinda similar, I decided not to use stain augmentations. I used multi scale training and scales are arranged between 1.5 and 2x raw scale i.e.

```python
train_scales = [
    (768, 768), (832, 832), (896, 896),
    (960, 960), (1024, 1024), (1088, 1088),
    (1152, 1152), (1216, 1216), (1280, 1280)
]
```
Other training augmentatios were random horizontal, vertical and diagonal flip, random brightness, contrast, hue and saturation.

Ayberk was using fixed size of 1024x1024 because we thought built-in augmentations in YOLO are already strong enough.

We used almost default values for optimizer and schedulers. I only added another learning rate multiplier step.

## Post-processing

Both of us thought post-processing was the most important in this competition because of the weird metric. We dedicated most of our time on this part rather than modelling.

First of all, we used 3x TTA (horizontal, vertical and diagonal flip) when we were predicting with our models. TTA predictions were merged using weighted boxes fusion. Masks that belong to fused boxes are averaged. Unfortunately, I couldn't find a way to retrieve soft mask predictions in mmdetection and I had to average binary masks. We didn't use multi scale TTA. YOLO model were taking 1024x1024 and Mask R-CNN models were taking 1280x1280 inputs.

Since I had 4 folds, I also had to merge their predictions and I used weighted boxes fusion with same configurations. We also tried to merge YOLO and Mask R-CNN predictions at this stage. It was working as good as other predictions on public leaderboard but always failed on private leaderboard for some reason.

We also tried the same submissions with and without dilation and we found that TTA was reducing the effect of dilation A LOT on public leaderboard. It was a really good sign because TTA is safe and dilation isn't.

We had different submissions with combinations of models, folds, TTA, dilation, score multiplication, WBF tuning and etc.

When we sort our submissions by public score, we get submission with dilation or ensemble.
[public]

When we sort our submission by private score, we get submissions with TTA, without dilation and single models but I also have one submission with dilation that scores 0.50x on private which is weird.
[private]

Our best submission was 14th place on private leaderboard but at the end Ayberk was able select a good enough submission that made us finish 30th.
