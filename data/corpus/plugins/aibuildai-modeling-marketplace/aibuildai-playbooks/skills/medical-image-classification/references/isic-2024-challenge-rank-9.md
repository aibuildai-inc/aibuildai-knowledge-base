# 9th Place Solution

Competition: isic-2024-challenge
Rank: #9
Source: https://www.kaggle.com/c/isic-2024-challenge/discussion/532577

Hi everyone,

I can't even explain how I am feeling right now?!? I finally made it to the gold zone for the first time after almost 3.5 years of Kaggling and made it as a solo participant. I am quite happy my solution performed robustly in the private LB too.

I would like to thank Kaggle and the organizers for creating this interesting challenge that will be beneficial and will improve the lives of many people worldwide. Also a big shout-out to the organizers who were extremely responsive during the competition which is appreciated a lot!

**TLDR:**
```
I built 2 (+1 from public) different tabular pipelines with various features, data, and model combinations paired with predictions from various image model backbones. Then I blended the results of these pipelines and made my final predictions. Weights for the different pipelines found via the hill-climbing algorithm.
```

I used a 5-fold StratifiedGroupKfold for each of the following experiments. For each feature I added, I tried them first with 5 different seed combinations. If they improved the score, I submitted it to the LB. And if they improved the score there as well, I kept that feature.

**Tabular Pipelines**

I am using the term pipeline here because each of the three pipelines uses different tabular models such as LGBM, Catboost, or XGBoost. In addition to the base features, I used the following features from my baseline  [notebook](https://www.kaggle.com/code/snnclsr/lgbm-baseline-with-new-features).

1. Pipeline 1
This is my original script where I tried to extract as many contextual features as possible from the patient data. I extracted the Z-score, range of the features, skewness, kurtosis, and feature / max(feature). I subsampled the negatives with a ratio of 0.02. I also added the mean of efficient net predictions as an image feature (described below). I had one LGBM and one Catboost here.
This pipeline gave 18.3 on its own in the public LB with a CV of 17.7.

2. Pipeline 2
With my own code, I was swinging between 10th to 20th place in the LB until @greysky 's [notebook](https://www.kaggle.com/code/greysky/isic-2024-only-tabular-data) came out. Quite luckily, we were using the same CV setup. I wanted to try how the blend would perform, so I added a couple of my features to that notebook such as
```python
        .with_columns(
            n_images_per_location = pl.col("isic_id").count().over(["patient_id", "tbp_lv_location_simple"])
        )
```
and my swin-transformer model's predictions as an image feature. On its own, this pipeline gave 17.5 CV and 18.3 LB. The blend with pipeline 1 gave my first significant boost of 18.6.

3. Pipeline 3
With the motivation of adding the third image model and making the ensemble a bit more diverse, I created another pipeline with a different feature setup. I created rank, and different groupby features on top of the previous features and used a different subset of the data (0.05 negative sampling ratio). This pipeline gave a 17.5 CV and 18.0 LB on its own.

**Image Models**

My main focus was diversity so for each of the pipelines, I chose a different backbone.

1. Two efficientnet_b0 with image size 256. LB: 15.5 and 15.8 with TTA 16.1
2. swinv2_tiny_window8_256 with image size 256. LB: 15.9
3. convnextv2_tiny.fcmae_ft_in22k_in1k with image size 224. LB: 15.8

I did a bunch of experiments with different data, backbone, and optimizer setups. The following setup gave me the most consistent results for both CV and LB.

I used only 5% of the negative images and upsampled the positives by 10. I trained only for 3 epochs as more epochs were making the results worse. I used constant lr:1e-4 for efficient-net and swin transformer models and used the following scheduler for the convnext model. For the optimizer, I just used Adam. For the loss function, I used BCE. 
```python
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)
```
For the augmentations, I used the following setup similar to the previous competition's winners:
```python
train_transforms = A.Compose([
    A.VerticalFlip(p=0.5),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(brightness_limit=0.2,contrast_limit=0.2, p=0.75),
    A.OneOf([
        A.MotionBlur(blur_limit=5),
        A.MedianBlur(blur_limit=5),
        A.GaussianBlur(blur_limit=5),
        A.GaussNoise(var_limit=(5.0, 30.0)),
    ], p=0.7),
    A.OneOf([
        A.OpticalDistortion(distort_limit=1.0),
        A.GridDistortion(num_steps=5, distort_limit=1.),
        A.ElasticTransform(alpha=3),
    ], p=0.7),
    A.CLAHE(clip_limit=4.0, p=0.7),
    A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.5),
    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, border_mode=0, p=0.85),
    A.CoarseDropout(p=0.7),
    A.Resize(CFG.img_size, CFG.img_size),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])
valid_transforms = A.Compose([
    A.Resize(CFG.img_size, CFG.img_size),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])
```
Something funny here at least for the people with a vision background, my image results were stuck at some point at 14.8, and whatever I tried did not work. While checking the code and looking for a solution for improvements, I realized that the `A.Resize` call was at the beginning of the list. While working on the image part of my solution I was not using any augmentations in the beginning and I was just appending new ones to the list as I experiment. Moving it to the end just before the normalization improved my scores to the 15.4-15.5 range.

**Final Blend**
To find the weights of the final models I used the hill-climbing algorithm. Thanks to @cdeotte for the implementation [here](https://www.kaggle.com/code/cdeotte/public-lb-1st-place-solution).
```
model	weight
0	pipe3_pred_lgb	0.407660
1	pipe3_pred_xgb	0.272537
2	pipe1_pred_lgb	0.152162
3	pipe1_pred_cat	0.142441
4	pipe_2_pred	        0.124199
5	pipe3_pred_cat	-0.099000
```
The CV of this setup was 18.2 and the LB was 18.7 which was my best CV and LB at the same time.

**Things did not work**
- I think similar to most of us, many many features
- For image models, I tried to use hard negatives where the models were making the highest errors, but it didn't work out.
- Focal loss performed always worse than BCE. Most probably because of my setup. 
- Mixup was a nice addition for the diversity but performed usually poorly when added to the ensemble.
- Stacking additional ExtraTreeClassifier/LogisticRegression model on top of my solution
- Scaling predictions with ** 0.5 or rank ensemble.
- Fixing the `tbv_lv_y`. There were 5 patients in the dataset and each of them was from the same hospital with negative `tbv_lv_y` value. I added the min per patient to fix it. The CV was slightly better but there was no improvement on the LB.
- Dullrazor algorithm for hair removal
- Hair augmentations
- Sample weights based on lesion id
- Training image models from scratch with dataset mean&std instead of imagenet init 

That was my summary. I will try to add more things in the upcoming days.
