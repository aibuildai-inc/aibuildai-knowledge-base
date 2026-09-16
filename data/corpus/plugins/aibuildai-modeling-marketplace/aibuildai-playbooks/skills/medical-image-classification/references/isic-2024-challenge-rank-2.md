# 2nd Place Solution

Competition: isic-2024-challenge
Rank: #2
Source: https://www.kaggle.com/c/isic-2024-challenge/discussion/532704

Hi Kagglers!!

First and foremost, we would like to express our sincere gratitude to the competition organizers and the community. Throughout the competition, we had the opportunity to explore various discussions, which served as a valuable source of inspiration for our ideas. This experience has been incredibly educational, allowing us to learn a great deal.

Personally, this was my first time collaborating with this particular group of team members. I was thoroughly impressed by their exceptional skills and found the experience to be highly stimulating.

## Overall Approach

The fundamental structure of our model aligns with a commonly used approach in public notebooks: incorporating image model features into tabular data, followed by inference using multiple GBDTs. 
We implemented several enhancements to both the GBDT and image models.

## GBDT Models

### Algorithms and Ensemble Strategy
- Used LGBM, XGBoost, and CatBoost
- Created 18 variations of each algorithm, resulting in a total of 54 models
- Employed seed averaging (n=5) using models trained on the full dataset

### Feature Engineering
For our base features, we adopted feature engineering techniques from the following public notebooks:
- [LGBM Baseline with New Features](https://www.kaggle.com/code/snnclsr/lgbm-baseline-with-new-features/notebook)
- [LightGBM CatBoost with New Features](https://www.kaggle.com/code/vyacheslavbolotin/lightgbm-catboost-with-new-features/notebook#Feature-Engineering)
- [ISIC 2024 LGBM ImageNet v5a](https://www.kaggle.com/code/richolson/isic-2024-lgbm-imagenet-v5a)

Additionally, we engineered several patient-related features to capture different aspects of the data:
- Patient-wise standardization
- Standardization by patient and tbp_lv_location
- Standardization by patient and tbp_lv_location_simple
- Standardization by patient and anatom_site_general
- Implemented the **Tabular Ugly Ducklings** technique (as described in this [discussion](https://www.kaggle.com/competitions/isic-2024-challenge/discussion/530183) in the competition forum)

To introduce diversity, some models used only a subset of these features.


### Image Features Integration
- Models used varying numbers of image features (0-3) as meta-features
- This variation in feature usage contributed to model diversity

### Hyperparameter Tuning
- Set `num_boost_round` between 200-300
- Conducted separate hyperparameter tuning for different combinations of:
  - Number of image features used
  - Number of patient features used

### Model Diversity
- Slightly varied the tabular features used across models
- Combined with the varying number of image features, this approach ensured a diverse ensemble of models


## Image Models

### Overview
We created a total of nine image models using five different training setups for diversity. Specifically, we integrated auxiliary losses for predicting tabular data and implemented self-supervised learning to enhance accuracy. Additionally, by selecting models with low variance across folds, we aimed for stable performance.

### Training Setups
1. **Standard Training**: Models were trained using basic configurations.
2. **Mixup Augmentation**: Mixup was added as a data augmentation technique during training.
3. **Auxiliary Loss for Predicting Tabular Data**: We introduced an auxiliary task for predicting tabular data to encourage learning from multiple modalities.
4. **Auxiliary Loss for Predicting `iddx_full` Clusters**: `iddx_full` was vectorized using tf-idf, followed by clustering via k-means. The model was trained to predict the distance from each data point to the cluster centroids as an auxiliary loss.
5. **Self-Supervised Pre-training with Tabular Data**: Following a recent multimodal learning paper [1], we conducted self-supervised pre-training with tabular data, then fine-tuned the image models.

### Image Models
The following nine models were trained with the respective setups and achieved the listed CV scores:

| Model | Training Setup | CV Score |
|----------------------|----------------------------------------|----------|
| **eva02_small** | Standard training | 0.1537 |
| **deit3_small** | Standard training | 0.1534 |
| **beitv2_base** | Mixup augmentation | 0.1594 |
| **convnextv2_tiny** | Auxiliary loss for tabular data | 0.1548 |
| **swinv2_small** | Auxiliary loss for `iddx_full` clusters| 0.1612 |
| **eva02_small** | Auxiliary loss for `iddx_full` clusters| 0.1580 |
| **resnext50** | Auxiliary loss for `iddx_full` clusters| 0.1515 |
| **convnextv2_nano** | Self-supervised pre-training with tabular data | 0.1607 |
| **swin_tiny** | Self-supervised pre-training with tabular data | 0.1596 |

### Common Training Configurations
- **Undersampling**: Each epoch applied undersampling at a ratio of 1:3 or 1:5.
- **Epoch Count**: Each model was trained for 50 to 200 epochs without early stopping to prevent overfitting to the validation set.
- **Data Augmentation**: Data augmentation strategies were adjusted based on the top solution from ISIC 2020 [2], with augmentation intensity varying depending on the model.
- **Optimizer**: AdamW was used with learning rates set to 1e-5 to 8e-6 for the backbone and 1e-3 for the head, alongside a warmup and cosine scheduler.

### Inference
- Models were trained on the full dataset and used for inference.
- Automatic Mixed Precision was enabled for faster inference.


## Cross-Validation Strategy

In addition to the Public LB, our team heavily relied on the results from this CV strategy for model evaluation and selection. For this competition, we implemented a **Triple Stratified Leak-Free KFold CV** strategy, inspired by an approach used in a previous Kaggle competition. This method ensures robust model validation while preventing data leakage.

The key aspects of this CV strategy are:

1. **Patient Isolation**: All images from a single patient are kept in the same fold, preventing leakage during cross-validation.
2. **Malignant Image Balance**: The stratification considers the proportion of malignant images for each patient.
3. **Patient Image Count Distribution**: Patients are binned based on their number of images, which is used for stratification.

We used a 5-fold Stratified Group KFold cross-validation for this competition, which implements all these aspects simultaneously.

For the original inspiration and more detailed explanation, refer to: [SIIM-ISIC Melanoma Classification - Triple Stratified CV](https://www.kaggle.com/c/siim-isic-melanoma-classification/discussion/165526)

## What Didn't Work Out
We attempted to incorporate data from past competitions. To align the tones, we applied techniques such as histogram matching, etc. However, unfortunately, this approach did not yield significant improvements in accuracy.
For further validation, we conducted an experiment where we mixed data from ISIC2018 with the current competition data. We then built an image model to distinguish between the past data and the current data. The results showed that the model achieved an AUC of 0.99 with relative ease.
Based on these results, we concluded that there must be distinct differences that we hadn't been able to identify visually. Consequently, we decided to forgo the use of past competition data in our approach.

## References

[1] Du, Siyi, Zheng, Shaoming, Wang, Yinsong, Bai, Wenjia, O'Regan, Declan P., and Qin, Chen. "TIP: Tabular-Image Pre-training for Multimodal Classification with Incomplete Data." In 18th European Conference on Computer Vision (ECCV 2024).

[2] [https://www.kaggle.com/competitions/siim-isic-melanoma-classification/discussion/175412](https://www.kaggle.com/competitions/siim-isic-melanoma-classification/discussion/175412)

## Code
https://github.com/uchiyama33/isic-2024-2nd-place
