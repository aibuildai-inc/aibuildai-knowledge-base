# 35th place solution

Competition: lish-moa
Rank: #35
Source: https://www.kaggle.com/c/lish-moa/discussion/200795

[DSG's 35th place solution]

## Notebooks

- Kon's ResNet: https://www.kaggle.com/yxohrxn/resnetclassifier-fit
- Blending: https://www.kaggle.com/yxohrxn/votingclassifier-fit-without-lightgm-and-svm
- Inference: https://www.kaggle.com/yxohrxn/votingclassifier-predict-without-postprocessing

## Our approach

### 1. Data augmentation

We tried Gaussian noise, Cutout, Mixup, and CutMix, and CutMix improved the CV score the most. In particular, it worked well when we increased the alpha, which is a hyperparameter. Removing ctl_vehicle made CutMix less effective.
 
### 2. Weighted seed-fold averaging

Since [Deotte's CV scheme](https://www.kaggle.com/c/lish-moa/discussion/195195) takes into account drug_id, a situation often arises where there are no positive examples in the training data, especially for minor classes. The predictions of the model created in this case are completely meaningless. Therefore, we counted the positive examples of each class in the training data for each fold and weighted the predictions using their number as weights.

### 3. Class-wise blending

We improved [Zhang's notebook](https://www.kaggle.com/gogo827jz/optimise-blending-weights-with-bonus-0) to compute weights to minimize the per-class log loss.

```python
weights.shape = (n_classes, n_models)
```

To find the best model combination, we evaluated the performance with CV. In addition, early stopping and label smoothing were performed to avoid overfitting. We created nearly 20 models in total, and the above combination was the best.
