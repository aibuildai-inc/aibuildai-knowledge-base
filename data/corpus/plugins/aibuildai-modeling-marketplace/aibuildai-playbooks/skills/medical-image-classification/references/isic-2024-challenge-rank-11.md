# 11th place solution

Competition: isic-2024-challenge
Rank: #11
Source: https://www.kaggle.com/c/isic-2024-challenge/discussion/532595

First, I would like to take the opportunity to thank The ISIC for hosting the competition and the Kaggle team for making it happen.

Here are details of the 11th place solution (CV:0.180 / Public:0.184 / Private:0.171).

# Cross-Validation Strategy

- 5-fold Stratified GroupKFold with 4 classes (benign, melanoma, SCC, and BCC)

# Tabular Model

- Almost the same as [this notebook](https://www.kaggle.com/code/greysky/isic-2024-only-tabular-data) (Thank you!).

# Image Model

- timm/convnext_small.fb_in22k
- timm/convnext_base.fb_in22k_ft_in1k
- timm/swinv2_cr_tiny_ns_224.sw_in1k
- Referenced [this notebook](https://www.kaggle.com/code/motono0223/isic-pytorch-training-baseline-image-only) (Thank you!).

|                   | convnext small | convnext base | swinv2 tiny |
|-------------------|----------------|---------------|-------------|
| learning rate            | 1e-4            | 1e-4           | 1e-4         |
| batch size            | 32            | 32           | 32         |
| training epochs (best pAUC selected)            | 5            | 5           | 5         |
| positive : negative            | 1:100            | 1:30           | 1:60         |
| dropout            | 0.2            | 0.0           | 0.2         |
| pooling            | Gem            | Gem           | AdaptiveAvgPool2d         |
| auxiliary loss (*1)            |             | ○           | ○         |
| external images (*2)            |             |            | ○         |
| CV            |      0.158       |      0.161      |     0.153     |
| Public            |      0.154       |     0.160       |     0.154     |
| Private            |      0.146       |      0.148      |    0.144      |

## auxiliary loss (*1)

- BCELoss(Binary: benign or malignant) + CrossEntropyLoss(Multiclass: benign, melanoma, SCC and BCC)

## external images (*2)

- Used external malignant images from [this dataset](https://www.kaggle.com/datasets/tomooinubushi/all-isic-data-20240629) (Thank you!).

- Excluded images with image_type "clinical: overview."

# Other

- Added prediction for each class from the multi-class learning as features.

- Scaled the predictions for each seed by dividing them with their max value.

- Changed the seed for undersampling. And ran an experiment using the average pAUC from 5 folds × 5 sampling patterns.

- For the submission, performed seed averaging (different undersampling seeds) with all training data.

Thank you!
