# [Solution] 22 Place, LB 0.650

Competition: imet-2019-fgvc6
Rank: #22
Source: https://www.kaggle.com/c/imet-2019-fgvc6/discussion/94783#latest-547886

Here are my findings and tricks, which helped me to get LB 0.650 by combining 3 models: se\_resnext101\_32x4d, pnasnet5large, senet154.

**Validation**: I used CV 5. Folds were made by the [iterative stratification package](https://github.com/trent-b/iterative-stratification). The package is extremely useful for unbalanced dataset and multiclass classification.

**Scheduler**: is one of the most crucial part for the fast convergence. I used 
CosineAnnealingWarmRestarts increasing the frequency by factor of two after each restart (see the sketch). Using this strategy 15 Epochs(3 restarts) were enough to converge. And I was able to test different ideas much faster.
[Learning rate]

**Augmentation:** everything is very standard: resize + random crop, small color jitter, hflip and small affine:
```
    transforms.Resize(img_size),
    transforms.RandomCrop((img_size, img_size), padding=0, pad_if_needed=True),
    transforms.ColorJitter(brightness=0.1,  contrast=0.1, saturation=0.1, hue=0.1),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomAffine(degrees = 10, translate= (0.0, 0.2), scale=(0.8, 1.2), shear=10)
```
**Mixup**: is a trick from the paper: [Bag of Tricks for Image Classification with Convolutional Neural Networks](https://arxiv.org/abs/1812.01187). The parameter of beta distribution was reduced by the same scheduler as the learning rate. It was reduced from 1 (very aggressive mixup) to 0 (no mixup at all). Mixup produced better validation results, but made the effect of TTA smaller. At the end it added around 0.002 per model. Adding this trick also made the convergence slower.

**Dropout**: The finding for the se\_resnext101\_32x4d: the standard dropout is 0.3, after increasing the dropout to 0.5 the CV and LB increased by ~0.005.

**Loss**: BCEWithLogitsLoss. The experiments with using Focal Loss or combining Focal Loss with BCE were not successful.

**Batch Size**: gradient accumulation didn’t help me much, so I just fit the batch size to occupy the whole GPU memory.

**Threshold fitting**: I guess it helped a lot to get nice results and made it easier to compare different experiments based on LB. 
First, I used validation to get the optimal threshold value, then I used this threshold to calculate average amount of labels per instance for the validation. It was equal to 5.2. During the inference I dynamically calculated the threshold such that I will approximately have 5.2 labels per image. Here is the code:

```
desired_mean = 5.2
for th in np.arange(1000)/10000. + 0.01:
    pred = (np.array(res) &gt; th).astype(np.float)
    if np.abs(pred.sum()/len(pred) - desired_mean) &lt; closest:
        closest = np.abs(pred.sum()/len(pred) - desired_mean)
        fix_th = th
```


I also have another strategy, where I try to have a certain distribution of the predictions. It performed just a bit better (&lt;0.001) than the previous strategy, but it is quite hard to explain (I can share the code if someone will be interested).


**TTA**: All augmentations from training, parameters of RandomAffine and ColorJitter are smaller.

**Results**:

| model | description | CV | LB |
| --- | --- | --- | --- |
| se\_resnext101\_32x4d | size: 300x300; + TTA4 | 0.607 | 0.642 |
| pnasnet5large | size: 331x331; + mixup + TTA4 | 0.609 | 0.641 |
| senet154 | size: 224x224; + mixup + TTA4 | 0.605 | 0.639 |


Ensemble of this 3 Networks with TTA2: 0.650.
