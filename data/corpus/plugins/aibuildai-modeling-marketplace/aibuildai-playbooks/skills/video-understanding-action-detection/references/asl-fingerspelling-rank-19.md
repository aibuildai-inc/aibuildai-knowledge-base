# 19th Place Solution - Jasper models

Competition: asl-fingerspelling
Rank: #19
Source: https://www.kaggle.com/c/asl-fingerspelling/discussion/434795

My solution is quite simple. I wasn't going to write about it, but there are no Jasper-like solutions in discussions yet :)

# Preprocessing 
Preprocessing was almost the same with [previous competition](https://www.kaggle.com/competitions/asl-signs/discussion/406306). But deleted mixup and replace augmentation. And add more complex time augmentations.

# Loss
I only used CTC loss.

# Data split
Random split **not** by user for 20 folds grouping by data length.
Deleted obviously bad samples:
```python
supplemental_metadata_df = supplemental_metadata_df[~((supplemental_metadata_df['length'] < 10) &
                                            (supplemental_metadata_df['phrase_len'] > 8))]

train_df = train_df[~((train_df['length'] < 7) & (train_df['phrase_len'] > 5))]
``` 
Used all supplemental data in train.

Also tried to create second loader with pseudo labels on bad predicted samples. Or delete bad predicted samples. Used both loaders while training:

```python
loader_to_use = 'main_loader'
if epoch % 3 == 0:
    loader_to_use = 'second_loader'
```
But it didn't help much.

# Model
First I tried simple quartznet, but with all strides set to 1.
Then I added some features step by step and trained each experiment from previous best checkpoint.

### Same for all runs:
* Optimizer: LookAheadAdamW
* Scheduler: Onecycle
* Batch size: 80

### List of experiments:
* Simple 5x5 quartznet with stride = 1.  [lr = 8e-3] [epochs=200] CV:0.8035
* Simple 5x5 quartznet with stride = 1. [lr = 7e-3] [epochs=150]  CV: 0.8106
* Previous + SE blocks.  [lr = 6e-3] [epochs=171]  CV: 0.8160
* Previous + Deep supervision (DSV) outputs for train loss. [lr = 5.8e-3] [epochs=175]  CV: 0.8211
* Previous without DSV + higher aug probabilities. Note: score worse but it needed to train one checkpoint without DSV for better next train results. [lr = 6e-3] [epochs=180]  CV: 0.8177
* Previous + DSV + increased dropout. [lr = 6.1e-3] [epochs=200]  CV: 0.8237
* Previous - DSV + lower dropout. [lr = 6e-3] [epochs=173]  CV: 0.8188
* Previous + Hyper Column (HYP) + 2 more blocks with 5 repeats. [lr=5.8e-3] [epochs=180] CV: 0.8252
* Previous + masked 1conv + masked SE + 2D DepthWise CNN layer before classifier + more dropout. [lr=5.8e-3] [epochs=200]  CV: 0.8294
* Previous + lstm after 2D Conv. [lr=1.75e-3] [epochs=125] CV: 0.8301


# What didn't work for me:
* Conformer models. I tried it but I guess without enough effort.
* Adding attention or conformer blocks to Jasper models.

When I used raw conformer and it has very low score. But when I changed input 2D conv to 1D conv or when I just deleted it - it was much better, but still little worse than my Jasper models.

# What I should have done, but didn't:
* Train more epochs
* Try better with conformer or squeezeformer
* Add post processing
* Try seq2seq
* Fixed length of input

Thanks to Kaggle for this great competition. Thanks to all top participants who shared their awesome solutions. 
Congratulations to all winners!
