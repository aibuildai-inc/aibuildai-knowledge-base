# 12th Solution

Competition: isic-2024-challenge
Rank: #12
Source: https://www.kaggle.com/c/isic-2024-challenge/discussion/532642

I would like to thank Kaggle for organizing such an interesting competition, and I would also like to express our gratitude to those who shared their amazing baseline notebooks.And big thanks to my teammates @stefanoclss , @alejopaullier , @yyyu54 and @kuixizhu  for trying their ideas.

# Approach Overview
We improved the baseline code([notebook 1](https://www.kaggle.com/code/motono0223/isic-tabular-model-image-model-features) and [notebook 2](https://www.kaggle.com/code/greysky/isic-2024-only-tabular-data) ) and used the best CV model and best LB model for the final submissions. However, both models had scores of public: 0.183 and private: 0.171..
## 1. Best LB Model (Public 0.183 / Private 0.171)
### Architecture:


**glcm_features :**Extract features from the images using GLCM (Gray Level Co-occurrence Matrix).
**KNN Feature :**The average of the numerical features from the closest (K = 5) data points for each patient, to find ugly duckling.
**Add Gaussian noise to train data :**Add noise to the oof prediction to prevent over-optimization of the CV score caused by the use of early stopping.
**Image Models :**
| No. |     Model Name     | CV Score | LB Score |
|:---:|:------------------:|:--------:|:--------:|
|  1  |   **EfficientViT-v2**  |  0.156   |  0.153   |
|  2  |    **EdgeNeXT-base**     |  0.156   |  0.155   |
|  3  |   **Efficient-B2**   | 0.1493   |  0.154   |
|  4  |   **Efficient-B0**   |  0.151   |  0.144   |
|  5  |     **EVE02**       |  0.154   |  0.154   |




### CV Strategy :
Triple Stratified Split([Reference](https://www.kaggle.com/competitions/siim-isic-melanoma-classification/discussion/165526))

## 2. Best CV Model (Public 0.183 / Private 0.171)

### Architecture:


We added the following processes to the Best LB Model
**age Feature :**Add some features that take advantage of differences in age and tbp_tile_type within the same patient.
**quartiles_feature :**Add Quartiles for numeric columns
**Feature selection :** Extract around 50 to 100 important features for CatBoost, LightGBM, and XGBoost respectively.Three models were trained with this feature.

### What didn't work
・Use of past data
・Pseudo-labeling
・FTTransformer (CV: 0.178, LB: 0.179, Private score: 0.165)

### P.S 
This competition held special meaning for me as my mother was diagnosed with cancer during the competition. I sincerely hope for her recovery and for medical breakthroughs that will lead to improved early detection of cancer for others
