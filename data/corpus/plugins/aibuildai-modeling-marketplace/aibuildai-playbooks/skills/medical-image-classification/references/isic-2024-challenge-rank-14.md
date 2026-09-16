# 14th place solution

Competition: isic-2024-challenge
Rank: #14
Source: https://www.kaggle.com/c/isic-2024-challenge/discussion/532786

First, I would like to sincerely thank the organizers of the ISIC 2024 Competition. I would also like to thank @daikon99 and @yuyagi for teaming up with me! On behalf of my team, I am excited to share our solution.

## Overview of the Inference Pipeline

We made improvements based on this excellent notebook ([ISIC 2024 Borrowed .179LB Tabular/OOF ImageNet](https://www.kaggle.com/code/richolson/isic-2024-borrowed-179lb-tabular-oof-imagenet)). In order to build a model that is robust against variance, we maximized the number of ensembles by increasing the diversity of the CNN models used for feature extraction. For the final submission, We chose the one based on Trust LB and the one based on Trust CV. As a result, the Trust LB Submission earned us a gold medal!

## Solution

### Validation Strategy

We developed our validation strategy with reference to the previous competition. ([Triple Stratified Leak-Free KFold CV](https://www.kaggle.com/competitions/siim-isic-melanoma-classification/discussion/165526))

To reduce the influence of patients with a large number of negative samples, we used the following method for fold splitting:
1. For patients with more than 100 negative samples, we limited the count to 100 negative samples per patient.
2. After applying this limit, we divided the samples from each patient ID in a way that ensured the number of positive samples was balanced across folds. 
3. Finally, for patient IDs where the number of negative samples was limited, we allocated all negative samples to the same folds as the corresponding patient ID.

In our experiments, the CNN and Stacking GBDT models trained using this split had a higher correlation with CV and Private LB than those trained using other splits (Grouped K-Fold or Stratified Grouped K-Fold based on patient ID). 
They also performed better on the Private LB than those trained using other splits. 

```python
neg_counts = 100

target_counts = train_df.groupby('patient_id')['target'].value_counts().unstack(fill_value=0).reset_index()

target_counts.columns = ['patient_id', 'target_0_count', 'target_1_count']
target_counts['total_images'] = target_counts['target_0_count'] + target_counts['target_1_count']
sorted_target_counts = target_counts.sort_values(by='total_images', ascending=False).reset_index(drop=True)

for patient_id in sorted_target_counts['patient_id']:
    target_0_max = sorted_target_counts.loc[sorted_target_counts['patient_id'] == patient_id, 'target_0_count'].values[0]
    if target_0_max >= neg_counts:
        patient_data = train_df[(train_df['patient_id'] == patient_id) & (train_df['target'] == 0)]
        if len(patient_data) > neg_counts:
            patient_data_sample = patient_data.sample(n=neg_counts, random_state=42)
            train_df = train_df[~((train_df['patient_id'] == patient_id) & (train_df['target'] == 0))]
            train_df = pd.concat([train_df, patient_data_sample])
```


### Inference Pipeline

**Result: CV→0.1793, Public LB→0.186 (2nd best), Private LB→0.171**

The final pipeline is a weighted average of three Stacking GBDT Models, GBDT Model without cnn prediction and single CNN Model, as shown below.


#### Stage 1. CNN with metadata

##### Input features

- Original features (ISIC2024 train_metadata.csv)
- Engineered features ([ISIC 2024 Borrowed .179LB Tabular/OOF ImageNet](https://www.kaggle.com/code/richolson/isic-2024-borrowed-179lb-tabular-oof-imagenet))

##### CNN Model Architecture

The CNN model was built based on the 1st solution ([1st place solution](https://www.kaggle.com/competitions/siim-isic-melanoma-classification/discussion/175412)) of the previous competition.
We modified model from this excellent [notebook](https://www.kaggle.com/code/motono0223/isic-pytorch-training-baseline-image-only) for the EfficientNetV2 model.

```python
# tf_efficientnet_b0_ns, swin_small_patch4_window7_224.ms_in21k_ft_in1k, convnext_tiny.in22k_ft_in1k 
class ISICModel(nn.Module):
    def __init__(self, model_name, n_meta_features,model_output_size, num_classes=1, pretrained=False,):
        super(ISICModel, self).__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained)
        
        self.model.classifier = nn.Identity()  

        self.meta_fc1 = nn.Linear(n_meta_features, 64)
        self.meta_fc2 = nn.Linear(64, 64)
    
        self.fc1 = nn.Linear(1128, 512) #　swins → 1128
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)

        n_meta_dim = [512, 128]
        self.meta = nn.Sequential(
                nn.Linear(n_meta_features, n_meta_dim[0]),
                nn.BatchNorm1d(n_meta_dim[0]),
                nn.Dropout(p=0.3),
                nn.Linear(n_meta_dim[0], n_meta_dim[1]),
                nn.BatchNorm1d(n_meta_dim[1]),
        )

    def forward(self, x):
        images, meta = x
        x1 = self.model(images) 
        x2 = self.meta(meta)
        x = torch.cat((x1, x2), dim=1) 
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# tf_efficientnetv2_s_in1k
class ISICModel(nn.Module):
    def __init__(self, model_name, num_classes=1, pretrained=True, config=None, meta_features=200):
        super(ISICModel, self).__init__()
        self.is_feature_only = True
        try:
            self.model = timm.create_model(
                model_name, in_chans=3, pretrained=pretrained, features_only=True
            )
            in_features = self.model.feature_info[-1]['num_chs']
        except Exception as e:
            print(e)
            self.model = timm.create_model(
                model_name, in_chans=3, num_classes=1, pretrained=pretrained)
            in_features = self.model.head.in_features
            self.model.head = nn.Identity()
            self.is_feature_only = False

        self.pooling = GeM()
        self.linear = nn.Linear(in_features + meta_features//2, num_classes)
        self.criterion = nn.BCEWithLogitsLoss()
        self.config = config
        self.training_step_outputs = []
        self.validation_step_outputs = []

        self.meta_bn = nn.BatchNorm1d(meta_features)
        self.meta_linear = nn.Linear(
            meta_features, meta_features//2)

    def forward(self, images, meta_data):
        if self.is_feature_only:
            features = self.model(images)[-1]
            if features.shape[1] == features.shape[2]:
                features = features.permute(0, 3, 1, 2)
        else:
            features = self.model.forward_features(images).unsqueeze(1)
            features = features.permute(0, 3, 1, 2)
        pooled_features = self.pooling(features).flatten(1)

        meta_data = self.meta_bn(meta_data)
        meta_data = self.meta_linear(meta_data)

        combined_features = torch.cat([pooled_features, meta_data], dim=1)

        output = self.linear(combined_features)
        return output
```

##### Augmentation

For augmentation, we based our approach on the 1st place solution ([1st place solution](https://www.kaggle.com/competitions/siim-isic-melanoma-classification/discussion/175412)) from the previous competition.

```python
# tf_efficientnet_b0_ns, swin_small_patch4_window7_224.ms_in21k_ft_in1k, convnext_tiny.in22k_ft_in1k 
A.Resize(CONFIG['img_size'], CONFIG['img_size']),
A.RandomRotate90(p=0.5),
A.Flip(p=0.5),
A.RandomBrightnessContrast(
    brightness_limit=0.2, contrast_limit=0.2, p=0.5),
A.OneOf([
    A.OpticalDistortion(distort_limit=1.0),
    A.GridDistortion(num_steps=5, distort_limit=1.),
    A.ElasticTransform(alpha=3),
], p=0.7),
A.Downscale(p=0.25),
A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.5),
A.ShiftScaleRotate(shift_limit=0.1,
                    scale_limit=0.15,
                    rotate_limit=60,
                    p=0.5),
A.CoarseDropout(max_height=int(CONFIG['img_size'] * 0.375),
                max_width=int(CONFIG['img_size'] * 0.375),
                max_holes=1, p=0.7),
A.Normalize()

# tf_efficientnetv2_s_in1k
A.Resize(CONFIG['img_size'], CONFIG['img_size']),
A.RandomRotate90(p=0.5),
A.Flip(p=0.5),
A.OneOf([
    A.OpticalDistortion(distort_limit=1.0),
    A.GridDistortion(num_steps=5, distort_limit=1.),
    A.ElasticTransform(alpha=3),
], p=0.5),
A.HueSaturationValue(hue_shift_limit=1, sat_shift_limit=1, val_shift_limit=1, p=0.5),
A.RandomBrightnessContrast(
    brightness_limit=0.2, contrast_limit=0.2, p=0.5),
A.Normalize()
```

##### CV & Public LB result of CNN model

| backbone | use original features | use engineered features | undersampling | lr | img_size | CV | Public LB |
| ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- |
| tf_efficientnet_b0_ns | ◯ | × | A maximum of 100 negative samples are used per patient | 3e-4 | 128 | 0.151 | 0.161 |
| swin_small_patch4_window7_224.ms_in21k_ft_in1k | ◯ | × | A maximum of 100 negative samples are used per patient | 1e-5 | 224 | 0.153 | 0.162 |
| convnext_tiny.in22k_ft_in1k | ◯ | × | A maximum of 100 negative samples are used per patient | 1e-4 | 288 |0.155 | 0.163 |
| tf_efficientnetv2_s_in1k | ◯ | ◯ | Random undersampling from all negative samples | 1e-4 | 224 | 0.165 | 0.170 |

#### Stage 2. Stacking GBDT and Ensemble

##### Input features

- Original features (ISIC2024 train_metadata.csv)
- Engineered features ([ISIC 2024 Borrowed .179LB Tabular/OOF ImageNet](https://www.kaggle.com/code/richolson/isic-2024-borrowed-179lb-tabular-oof-imagenet))
- Norm2 (aggregated features)
- CNN Predictions

##### Feature selection

Using the ideas from the [notebook](https://www.kaggle.com/code/murashow/tabular-with-image-features-lightgbm/notebook), calculated feature importance for each Stacking model, and removed unnecessary features with an importance of 0.0. As a result, both the CV and LB scores were improved.

##### Norm2: aggregated features (patient_id + tbp_lv_location_simple)

In addition to the base notebook features, add the following aggregated features using patient_id and tbp_lv_location_simple.

```python
pl.read_csv(path)
.with_columns(
    (pl.col('patient_id') + '_' + pl.col('tbp_lv_location_simple')).alias('patient_id_location')
)
.with_columns(
    ((pl.col(col) - pl.col(col).mean().over('patient_id_location')) / (pl.col(col).std().over('patient_id_location') + err)).alias(f'{col}_patient_location_norm') for col in (num_cols + new_num_cols)
)
```

##### CV results for the models used in the ensemble

For the stacking model, there was a proposal to either increase the number of models used as features or to increase the number of GBDTs. As a result of experiments, limiting the number of models used as features to a maximum of 2, while increasing the number of GBDTs, produced better results for both CV and LB.
In addition, tf_efficientnetv2_s_in1k had the highest scores in CV and Public LB, so we adopted it as an ensemble model. 

| model name | oof_1 | oof_2 | use norm2 | ensemble weight | CV |
| ---- | ---- | ---- | ---- | ---- | ---- |
| GBDT_1(lgb+cb+xgb) | tf_efficientnet_b0_ns |swin_small_patch4_window7_224.ms_in21k_ft_in1k | × | 1.882e-01 | 0.178 |
| GBDT_2(lgb+cb+xgb) | tf_efficientnet_b0_ns |swin_small_patch4_window7_224.ms_in21k_ft_in1k | ◯ | 4.963e-01| 0.179 |
| GBDT_3(lgb+cb+xgb) | convnext_tiny.in22k_ft_in1k | tf_efficientnetv2_s_in1k | ◯ | 1.534e-01 | 0.180 |
| GBDT_4(lgb+cb+xgb) | × | × | ◯ | 5.075e-02 | 0.171 |
| tf_efficientnetv2_s_in1k only | × | × | × | 2.702e-04| 0.165 |

##### Ensemble Method

Our ensemble method uses a weighted average, where the weights are calculated to minimize the loss.

```python
from scipy.optimize import minimize

def ensemble_loss(weights):
    weights = np.array(weights)
    weights = weights / np.sum(weights)
    
    ensemble_preds = weights[0] * oof_1['oof_pred'] + weights[1] * oof_2['oof_pred'] + weights[2] * oof_3['oof_pred'] + weights[3] * oof_4['oof_pred'] + weights[4] * oof_5['oof_pred']
    # calculate loss
    return 1 - comp_score(df_train["target"], ensemble_preds)

result = minimize(
    ensemble_loss, 
    [1/5, 1/5, 1/5, 1/5, 1/5], 
    method='Nelder-Mead',
    )
print(1-result.fun)
print(result)
```

### What Didn't Work
- Add features that only exist in the training data as prediction targets for the CNN.
- Soft labels if biopsied
- Post processing of age_approx
- TTA
- Fold split of unknown hospitals in the validation
- Add 3 or more oof features to a single GBDT
- Add Hair Augmentation
- Change to Focal loss
- Pre-train using past competition data
- Rank ensemble
