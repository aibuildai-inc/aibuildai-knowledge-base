# 15th place solution

Competition: cmi-detect-behavior-with-sensor-data
Rank: #15
Source: https://www.kaggle.com/c/cmi-detect-behavior-with-sensor-data/writeups/15th-place-solution

Congratulations to all the winners! I would also like to express my gratitude to the organizers for preparing such an exciting competition.
I devoted a significant amount of time and effort to this competition from the very beginning, so I am truly delighted that this has resulted in a gold medal.

# Overall Summary

The key factors in my approach were (1) building diverse models and ensembling them, and (2) applying post-processing by leveraging previously inferred test samples. The idea of creating diverse models was inspired by my past gold medal solution in the [HMS competition](https://www.kaggle.com/competitions/hms-harmful-brain-activity-classification). Since HMS also dealt with time-series sensor data, some of the model ideas in this competition came directly from that experience.

# Models

## Common Settings Across All Models

* Trained separate models for **IMU only** and **IMU + TOF + THM (All)**.
* Trained five different models (i–v) and ensembled them via weighted averaging.
* Used publicly shared IMU feature sets.
* 18-class output.
* Employed standard **Cross-Entropy loss**.
* Augmentation methods varied slightly by model, but **mixup** was consistently effective (except for plot-2DCNN).
* For sequences longer than 224, used the last 224 timesteps; for shorter sequences, applied padding.
* Preprocessing: applied **mean-std normalization** (except for plot-2DCNN, which used fixed-value clipping).
* StratifiedGroupKFold (5-folds)

## i) raw-2DCNN

This idea was introduced by a teammate in the HMS competition and achieved the best performance among my five models. The input images for the 2DCNN are as follows. All signals except TOF were stacked into 16 pixels, while TOF was left unstacked. Upscaling the width to double (448 pixels in width) further improved the score. The model architecture is EfficientNetV2-B2 (B1 for IMU-only).



## ii) plot-2DCNN

This was the idea I personally used in HMS. The input images for the 2DCNN are as follows. TOF part processed the same as in raw-2DCNN. The model architecture is EfficientNetV2-B1.



## iii) 1DCNN-GRU-Attention

A lightly modified version of a public notebook model.

## iv) 1DCNN-BERT

Same as iii), except the GRU-Attention module was replaced with a BERT.

## v) Mix of 1DCNN & Transformer

Based on [models used in past competition solutions](https://www.kaggle.com/competitions/asl-signs/writeups/hoyeol-sohn-1st-place-solution-1dcnn-combined-with), but adjusted with smaller dim, ksize, and fewer layers.

# Post-processing

Two days before the competition deadline, I discovered this technique and achieved a significant score boost (\~+0.010). The basic idea is similar to the one discussed in [this thread](https://www.kaggle.com/competitions/cmi-detect-behavior-with-sensor-data/discussion/603544).
I modified the `prediction()` function in the submission code into a class method, storing the predictions of previously inferred samples in `self.history` variable. This allowed the model to access past predictions. For each subject ID, I counted predictions per class, then penalized predictions belonging to classes with high counts.

# CV & LB

|Model|IMU only CV|All CV|Public LB|Private LB|
| --- | --- | --- | --- | --- |
|raw-2DCNN|0.7972|0.8773|-|-|
|plot-2DCNN|0.7695|0.8632|-|-|
|1DCNN-GRU-Attention|0.7999|0.8437|-|-|
|1DCNN-BERT|0.7834|0.8385|-|-|
|Mix of 1DCNN & Transformer|0.7883|0.8282|-|-|
|5 Models Ensemble|0.8211|0.8945|0.868|0.846|
|Ensemble + Post-process|-|0.9044|0.884|0.858|
