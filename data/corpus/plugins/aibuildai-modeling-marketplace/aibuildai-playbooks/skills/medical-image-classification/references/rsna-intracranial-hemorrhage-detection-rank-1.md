# 1st Place Solution. Sequential model wins

Competition: rsna-intracranial-hemorrhage-detection
Rank: #1
Source: https://www.kaggle.com/c/rsna-intracranial-hemorrhage-detection/discussion/117210

The key module of our pipeline is a sequence model. It works well and there is no shakeup.
Code : https://github.com/SeuTao/RSNA2019_1st_place_solution

# Overview


# 2D CNN Modeling

**Data pre-processing &amp; augmentation**
Our team has three 2d classifier pipelines. The three pipelines share different input settings (3 channels):
```
1. Single sclice with 3 windows.
2. Spatially adjacent 3 slices with one window.
3. Combination of 1 and 2: Spatially adjacent 3 slices with three windows.
```


The windows we use are:
```
Brain Window[40, 80],
Subdural Window[80, 200],
Bone Window[600, 2800]
```


Augmentations:
- Random ShiftScaleRotate
- Random resize crop 
- Random HFlip

Training strategy
- Randomly sample images form different SeriesInstanceUID
- Each epoch was trained on 4 times SeriesInstanceUIDs
- Adam optimiser with cycle learning rate (5e-4~1e-5)

# Sequence Model Development

**Sequence model 1:  MLP + LSTM** 
Input: 
- Slice embeddings from multi models (num_models*feature dim) 



**Sequence model 2:  1d CNN + LSTM**
Input: 
- Logits from multi 2D CNN models (num_models*6 class output) 
- Logits from sequence model 1 (6 class output) 
- Meta info (Position)


