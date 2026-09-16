# 10th Place Solution

Competition: isic-2024-challenge
Rank: #10
Source: https://www.kaggle.com/c/isic-2024-challenge/discussion/533179

Thank you to those who organized the competition!
Code: https://github.com/tomo20180402/isic2024_skin-cancer-detection-with-3d-tbp_10th-place-solution

# Overview

- **CV=0.183, Public LB=0.181, Private LB=0.171**
- Two-stage model
    - 1st stage: CNN (EfficientNet-B0), using only image as input
    - 2nd stage: GBDT (LightGBM, XGBoost, CatBoost), using predictions from 1st stage and tabular data as input
- The core of the solution is strategies for improving model generalization.

# Strategies for improving model generalization
Basically, trust CV.
Even if CV slightly worsens, I applied techniques if I thought they would improve generalization performance.
1. Repeat undersampling
    - Divide negative samples into several datasets for training
        - In the 1st stage, divide into 20 datasets and switch for each epoch (n_epoch=20)
        - In the 2nd stage, divide into 10 datasets and build models for each
    - Use multiple validation sets
        - StratifiedGroupKFold (group=patient, k=5) × 5 split seeds
2. Avoid early stopping
    - Investigate hyperparameters such as the number of epochs using several validations and fix them
    - Prevent accidentally getting good CV on specific folds (prevent overfitting to CV)
3. Deal with unknown categories
    - In the 2nd stage, add data with missing attributions to the training data
    - Treat unknown categories as missing during inference
4. Add diversity to models
    - Use 2 different augmentation patterns in the 1st stage
    - Use 3 types of GBDT (LightGBM, XGBoost, CatBoost) in the 2nd stage
    - Features of focus change depending on the model
5. Simple average ensemble
    - Average without weighting specific models
6. Prevent fitting to patients with many cases
    - Train up to 50 negative samples per patient
    - Included this 50% in the ensemble as a second submission, but it worsened except for public LB
        - cv=0.182, public LB=0.182, private LB=0.170

# 1st stage
- Model
    - EfficientNet-B0
- Augmentation
    - Pattern 1: [Winning solution of SIIM-ISIC Melanoma Classification](https://www.kaggle.com/competitions/siim-isic-melanoma-classification/discussion/175412), image_size=128, CV=0.152 (selecting the best epoch yields CV=0.155)
    - Pattern 2: CV=0.156 (selecting the best epoch yields CV=0.159)
- Total models
    - 2 augmentations × 5 split seeds × 5 folds = 50
```python
# Pattern 1
train_transform = A.Compose([
    A.Transpose(p=0.5),
    A.VerticalFlip(p=0.5),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightness(limit=0.2, p=0.75),
    A.RandomContrast(limit=0.2, p=0.75),
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
    A.Resize(128, 128),
    A.Cutout(max_h_size=int(128 * 0.375), max_w_size=int(128 * 0.375), num_holes=1, p=0.7),    
    A.Normalize()
])
valid_transform = A.Compose([
    A.Resize(128, 128),
    A.Normalize()
])

# Pattern 2
train_transform = transforms.Compose([
    transforms.Resize((144, 144)),
    transforms.RandomResizedCrop(128, scale=(0.8, 1.2), ratio=(0.75, 1.3333)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    RandomApply(transforms.RandomRotation(45), p=0.5),
    RandomApply(transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0, hue=0), p=0.5),
    RandomApply(transforms.GaussianBlur(kernel_size=3), p=0.3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
valid_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
```

# 2nd stage
- Model
    - LightGBM, XGBoost, CatBoost
- Feature
    - Pattern 1
        - 1st stage predictions
            - During actual inference, take the average of 1st stage predictions for each pattern as input for 2nd stage
        - features from public notebook, [ISIC 2024 | Only Tabular Data](https://www.kaggle.com/code/greysky/isic-2024-only-tabular-data?scriptVersionId=191634832)
        - standardized and count-based features of patient × other category
            - other category: ['tbp_tile_type', 'tbp_lv_location', 'tbp_lv_location_simple', 'attribution']
        - remove: ['tbp_lv_norm_color', 'tbp_lv_symm_2axis', 'tbp_lv_symm_2axis_angle', 'tbp_lv_nevi_confidence', 'sex', 'anatom_site_general']
        - some other features, see below for details
    - Pattern 2
        - The same as pattern 1 except for the following:
            - remove: ['tbp_lv_norm_color_per_long_diam', 'norm_A', 'norm_B', 'norm_L', 'clin_size_long_diam_mm_per_age', 'area_ratio', 'weighted_area_ratio']
- Total models
    - 2 features × 3 GBDTs × 5 split seeds × 5 folds × 10 sets = 1500
```python
...
# some other features
).with_columns(
    area_age_ratio = pl.col('tbp_lv_areaMM2') / pl.col('age_approx'),
    symmetry_eccentricity_interaction = pl.col('tbp_lv_symm_2axis') * pl.col('tbp_lv_eccentricity'),
    border_complexity_normalized = pl.col('tbp_lv_norm_border') / (pl.col('tbp_lv_areaMM2').sqrt() + err),
    color_variation_index = pl.col('tbp_lv_color_std_mean') * pl.col('tbp_lv_radial_color_std_max'),
    lesion_flatness = (pl.col('tbp_lv_x')**2 + pl.col('tbp_lv_y')**2).sqrt() / (pl.col('tbp_lv_z')+ err),
    relative_hue_difference = (pl.col('tbp_lv_H') - pl.col('tbp_lv_Hext')).abs() / (pl.col('tbp_lv_H') + pl.col('tbp_lv_Hext') + err),
    area_diameter_ratio = pl.col('tbp_lv_areaMM2') / (pl.col('clin_size_long_diam_mm')**2 + err),
    color_geometric_mean = (pl.col('tbp_lv_L') * pl.col('tbp_lv_A') * pl.col('tbp_lv_B')) ** (1/3),
    tbp_lv_norm_color_per_long_diam = pl.col('tbp_lv_norm_color') / pl.col('clin_size_long_diam_mm').log1p(),
    norm_L = (pl.col('tbp_lv_stdL') + pl.col('tbp_lv_color_std_mean')) / pl.col('tbp_lv_L'),
    norm_A = (pl.col('tbp_lv_stdL') + pl.col('tbp_lv_color_std_mean')) / abs((pl.col('tbp_lv_A')) + err),
    norm_B = (pl.col('tbp_lv_stdL') + pl.col('tbp_lv_color_std_mean')) / abs((pl.col('tbp_lv_B')) + err),
    clin_size_long_diam_mm_per_age = pl.col('clin_size_long_diam_mm') / pl.col('age_approx'),
    area_ratio = (np.pi * (pl.col('clin_size_long_diam_mm')/2)**2) / pl.col('tbp_lv_areaMM2'),
).with_columns(
    weighted_area_ratio = (pl.col('area_ratio') * (pl.col('tbp_lv_areaMM2'))).log1p(),
)
...
```
