# 4th Place Solution

Competition: isic-2024-challenge
Rank: #4
Source: https://www.kaggle.com/c/isic-2024-challenge/discussion/532760

Congratulations to all the winners! Thanks to Kaggle and ISIC for organizing this competition and introducing us this interesting problem. It's been over a year since I've been back to kaggle, and I am very happy to win this competition.
Special thanks to @greysky for providing the [amazing tabular notebook](https://www.kaggle.com/code/greysky/isic-2024-only-tabular-data/notebook), and I used your features in my solution.

## Code
Please give me a star if you find it useful ! 😀 https://github.com/dungnb1333/ISIC-2024 
## Inference kernel
- [4th place leaderboard prize](https://www.kaggle.com/code/nguyenbadung/isic-2024-final-submission?scriptVersionId=195231736)
- [Top-15 retrieval sensitivity](https://www.kaggle.com/code/nguyenbadung/isic-2024-secondary-prize?scriptVersionId=195319448)

My solution is described below

## Image pipelines
**Dataset**
        Isic 2024 from this challenge
        Isic 2019: https://challenge.isic-archive.com/data/#2019
        Isic 2020: https://challenge.isic-archive.com/data/#2020
        Isic 2018 for segmentation task: https://challenge.isic-archive.com/data/#2018
        PAD UFES 20: https://data.mendeley.com/datasets/zr7vgbcyr2/1  thanks @hengck23
   
**Multi-label classification**
        I feel that more granular classes will create better feature representations that can help improve performance. I used 5 classes target, MEL, BCC, SCC, NV for training, and I used the sigmoid probability of the target class for prediction. To map the ISIC 2019, 2020, 2024 and PAD UFES labels to the above 5 classes, I used the following rule:
```python
2019 MEL, BCC, SCC, NV -> MEL, BCC, SCC, NV
2020 melanoma -> MEL
2020 nevus -> NV
PAD UFES 20 MEL, BCC, SCC, NEV -> MEL, BCC, SCC, NV
2024 Basal cell carcinoma in iddx_full metadata -> BCC
2024 Melanoma in iddx_full metadata -> MEL
2024 Squamous cell carcinoma in iddx_full metadata -> SCC
2024 Nevus in iddx_full metadata -> NV
```
Target is set to 1 if one of the classes (MEL, BCC, SCC) is 1.

**Models**

**For 4th place leaderboard prize**

- 4 multi-label classification models (5 classes) trained with ISIC 2024+2020+2019 + PAD UFES, validated with ISIC 2024
swin_tiny image size = 224: OOF score = 0.1609
convnextv2_base image size = 128: OOF score = 0.1641
convnextv2_large image size = 64: OOF score = 0.1642
coatnet_rmlp_1 image size = 224: OOF score = 0.161
- 3 multi-task segmentation + classification models trained with ISIC 2024+2020+2019 + PAD UFES, validated with ISIC 2024. For the submission, I only used the prediction from the classification task
efficientnet-b3 Unet image size = 224: OOF score = 0.1638
mit-b0 FPN image size = 384: OOF score = 0.1671
mit-b5 FPN image size = 224: OOF score = 0.1656
      
- To create masks for the segmentation task, I trained 3 models with ISIC 2018 data and made predictions with ISIC 2024+2020+2019 + PAD UFES
efficientnet-b7 Unet++ image size = 256: IoU = 0.829
efficientnet-b5 Unet++ image size = 512: IoU = 0.827
mit-b5 FPN image size = 512: IoU = 0.843

- 3 multi-label classification models (5 classes) trained only with ISIC 2024 data
vit_tiny image size = 384: OOF score = 0.1688
swin_tiny image size = 256: OOF score = 0.1655
convnextv2_tiny image size = 288: OOF score = 0.1645

- Ensemble 10models: OOF score = 0.17458

**For Top-15 retrieval sensitivity prize**

- 1 multi-label classification models (5 classes) trained with ISIC 2024+2020+2019 + PAD UFES, validated with ISIC 2024
vit_tiny image size = 224: OOF score = 0.16040
- 1 multi-task segmentation + classification models trained with ISIC 2024+2020+2019 + PAD UFES
mit-b0 FPN image size = 224: OOF score = 0.1660

**Image augmentation**

```python
transform_train = albu.Compose([
    albu.Resize(self.image_size, self.image_size),
    albu.ImageCompression(quality_lower=80, quality_upper=100, p=0.25),
    albu.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, border_mode=0, p=0.5),
    albu.Flip(p=0.5),
    albu.RandomRotate90(p=0.5),
    albu.OneOf([
        albu.MotionBlur(blur_limit=5),
        albu.MedianBlur(blur_limit=5),
        albu.GaussianBlur(blur_limit=5),
        albu.GaussNoise(var_limit=(5.0, 30.0)),
    ], p=0.5),
    albu.RandomBrightnessContrast(p=0.5),
    albu.CoarseDropout(num_holes_range=(1,1), hole_height_range=(8, 32), hole_width_range=(8, 32), p=0.25),
    albu.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])
transform_val = albu.Compose([
    albu.Resize(self.image_size, self.image_size),
    albu.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])
```
**Exponential Moving Average:**
For submission, I use ema checkpoint (train with full data - no valid) at epoch 8->15, no TTA.

##  Tabular pipelines
I used 3 models: LightGBM, CatBoost, XGBoost. I combined 10 features from the image pipeline and all the features from the [amazing tabular notebook](https://www.kaggle.com/code/greysky/isic-2024-only-tabular-data/notebook). Thanks again, @greysky!

- Params:
```python
lgbm_params = {
    'objective':        'binary',
    'verbosity':        -1,
    'n_estimators':     300,
    'early_stopping_rounds': 50,
    'metric': 'custom',
    'boosting_type':    'gbdt',
    'lambda_l1':        0.08758718919397321, 
    'lambda_l2':        0.0039689175176025465, 
    'learning_rate':    0.03231007103195577, 
    'max_depth':        4, 
    'num_leaves':       128, 
    'colsample_bytree': 0.8329551585827726, 
    'colsample_bynode': 0.4025961355653304, 
    'bagging_fraction': 0.7738954452473223, 
    'bagging_freq':     4, 
    'min_data_in_leaf': 85, 
    'scale_pos_weight': 2.7984184778875543,
    "device": "gpu"
}
```

```python
cb_params = {
    'loss_function':     'Logloss',
    'iterations':        300,
    'early_stopping_rounds': 50,
    'verbose':           False,
    'max_depth':         7, 
    'learning_rate':     0.06936242010150652, 
    'scale_pos_weight':  2.6149345838209532, 
    'l2_leaf_reg':       6.216113851699493,
    'min_data_in_leaf':  24,
    'cat_features':      cat_cols,
    "task_type": "CPU",
}

```
```python
xgb_params = {
    'enable_categorical':       True,
    'tree_method':              'hist',
    'disable_default_eval_metric': 1,
    'n_estimators':             300,
    'early_stopping_rounds':    50,
    'learning_rate':            0.08501257473292347, 
    'lambda':                   8.879624125465703, 
    'alpha':                    0.6779926606782505, 
    'max_depth':                6, 
    'subsample':                0.6012681388711075, 
    'colsample_bytree':         0.8437772277074493, 
    'colsample_bylevel':        0.5476090898823716, 
    'colsample_bynode':         0.9928601203635129, 
    'scale_pos_weight':         3.29440313334688,
    "device":                   "cuda",
}
```

|                  | LGB pAUC | CB pAUC | XGB pAUC | Gmean pAUC |
| :--------------- | :------- | :------ | :------- | :--------- |
| [meta feat](https://www.kaggle.com/datasets/nguyenbadung/isic-2024-4th-place-weights?select=tab_meta_feat)    | 0.17806  | 0.17498 | 0.17954  | 0.17879    |
| 10model feat | 0.18650  | 0.18669 | 0.18647  | 0.18703    | 
| 2model feat | 0.18406  | 0.18378 | 0.18328  | 0.18438    | 

- For each model, I tried with 40 different seed combinations and blend top 5 checkpoints with the highest CV score for submission
 
## Final submission
|                                                    | PublicLB | PrivateLB | Prize |
| :------------------------------------------------- | :------- | :-------- | :-------- |
| [Sub1-GPU: 0.2*(meta only) + 0.8*(10model feat)](https://www.kaggle.com/code/nguyenbadung/isic-2024-final-submission?scriptVersionId=195231736) | 0.18229    | 0.17225     | 4th place leaderboard prize |
| [Sub2-CPU: 0.2*(meta only) + 0.8*(2model feat)](https://www.kaggle.com/code/nguyenbadung/isic-2024-secondary-prize?scriptVersionId=195319448)  | 0.18094    | 0.17011     | Top-15 retrieval sensitivity prize | 

That was my summary. I have a submission that reached private LB = 0.1732 (1st place) by changing the selection of image models, but it wasn't my best CV score.
