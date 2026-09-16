# 21st place solution

Competition: isic-2024-challenge
Rank: #21
Source: https://www.kaggle.com/c/isic-2024-challenge/discussion/532723

Thanks to the organizers for hosting this competition and to everyone who shared ideas and notebooks! Considering the small number of positive samples, I was expecting some sort of shakeup to happen, but the jump in the private leaderboard took me a bit by surprise. I picked one submission in the best CV and one in the best LB (which has a decent CV-LB correlation).

**Approach**

My approach is a stacked model, where class predictions from an image classifier are fed as a feature to boosting models such as xgboost, catboost and lgbm. Eventually, a weighted average of these 3 models is used as output. The weights are found using the best results for their oof predictions.

**CV**

I used GroupKFold(5) for cross-validation grouped by patient ids.

**Feature engineering and selection**

I used the features shared in most of the publicly shared notebooks. Thanks to everyone who shared. I performed feature selection because there were many correlated features. I filtered features that had > 0.8 correlation with any other feature and eventually was left with 116 features.

**Image classifier**

I experimented with a couple of image classifiers such as resnet, efficienet and its variants but observed better results with swin model from timm. I froze 39 initial layers of swin model and used lr=2e-5. Image size was 224x224 and used augmentations listed below along with early stopping with patience=2. For each fold training data, I used 1:5 ratio of positive to negative samples. I used focal loss(alpha=0.2, gamma=2) to handle class imbalance. I used normalization by dividing the image with 255 instead of using the imagenet mean and std.

```python
A.Compose([
        A.Resize(CONFIG['img_size'], CONFIG['img_size']),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Rotate(limit=90, p=0.5),
        A.CoarseDropout(max_holes=5, max_height=20, max_width=20, min_holes=1, min_height=10, min_width=10, p=0.3),
        A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.2),
        A.MotionBlur(blur_limit=5, p=0.3),
        A.MedianBlur(blur_limit=5, p=0.3),
        A.GaussianBlur(blur_limit=5, p=0.3),
        ToTensorV2()], p=1.),
```

**Boosting models**

I used 3 models xgboost, catboost and lgbm. For each fold, I used the balanced sampling approach shared [here](https://www.kaggle.com/code/richolson/isic-2024-imagenet-lr-ramp-target-mods) in order to combat the imbalance in positive and negative samples. I used the swin model's oof with a bit of gaussian noise to it while training the boosting models to prevent over reliance on its predictions. I performed hyperparameter optimization of the boosting models using optuna.

I also made to ensure that the folds remain the same for the image classifier and the boosting models to prevent information leakage.

Notebook: https://www.kaggle.com/code/rushali2406/21st-place-solution-isic?scriptVersionId=194724691
