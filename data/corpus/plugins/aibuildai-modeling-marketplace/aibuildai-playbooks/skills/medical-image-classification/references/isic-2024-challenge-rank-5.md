# 5th Place Solution

Competition: isic-2024-challenge
Rank: #5
Source: https://www.kaggle.com/c/isic-2024-challenge/discussion/533056

First of all, I just wanted to say a big thank you to the Kaggle Team and ISIC for hosting such a fantastic competition. It was a great learning experience and a lot of fun\!

## TLDR:

- Modeling of images and tables with and without data from past competitions  
- Model partitioning with known and unknown attribution  
- Trust CV

## Code
- train
    - [Tabuler Model 1 & Image Model 1](https://github.com/NYNYNY67/kaggle_isic)
    - [Tabuler Model 2](https://github.com/fyk7/kaggle-isic2024-5th-table-part-)
    - [Tabuler Model 3](https://github.com/auccchi/kaggle_isic)
    - [Image Model 2](https://github.com/ToseiHatori/isic)
- inference 
    - [inference notebook](https://www.kaggle.com/code/fyk778/merged-sub-final-ensemble-v1-5th-place)

## Validation Strategy

We used StratifiedGroupKFold with Patient\_id. Many public notebooks had leaks due to different splits between images and tables, so it was not an accurate CV.

On the other hand, it was specified that an unknown hospital was to be included in the test set( [discussion](https://www.kaggle.com/competitions/isic-2024-challenge/discussion/517139)), but this split method makes it impossible to simulate it. Therefore, the performance for the unknown attribution was checked with public LB.

## Models

### Summary



This image is a summary of our solution.  
More details are given below.

### Image Models

- Overview  
    - We created two models: one that used past data and another that used only the data from this competition.  
- Image Models 1 (Without past data  
    - **Dataset**  
        - no additional data  
        - down sample negative examples to the size of positive examples  
        - re-sample negative examples on every epochs  
  - **Training Setups** 
      - 30 epochs (no early stopping).  
      - No auxiliary loss  
      - backbone models  
          - convnext\_base.fb\_in22k\_ft\_in1k  
          - eva02\_small\_patch14\_336.min\_in22k\_ft\_in1k  
    - augmentation  
        - Resize, RandomRotate90, ShiftScaleRotate, HueSaturationValue, RandomBrightnessContrast  
    - optimizer  
        - AdamW  
        - lr: 1e-5  
    - scheduler  
        - CosineLRScheduler  
    - batch\_size  
        - 32  
    - instead of using 5 fold image models, single model was trained using full data for the sake of reducing inference time  

- Image Models 2 (With past data  
    - **Dataset**  
        - StratifiedGroup 5 Fold with Patient\_id (same as the table models)  
        - past data  
            - Both train and **validation** used past competition data  
                - [https://www.kaggle.com/competitions/siim-isic-melanoma-classification/discussion/175412](https://www.kaggle.com/competitions/siim-isic-melanoma-classification/discussion/175412)  
                - Validation loss was unstable when using only the data from this competition.  
            - dataset  
                - [https://www.kaggle.com/datasets/tomooinubushi/all-isic-data-20240629](https://www.kaggle.com/datasets/tomooinubushi/all-isic-data-20240629) thanks to @tomooinubushi  
            - target  
                - benign, indeterminate, indeterminate/benign are mapped to 0  
                - indeterminate/malignant, malignant are mapped to 1  
        - No downsampling or upsampling; used the entire dataset.  
  - **Training setups**  
      - 5 epochs (no early stopping).  
      - Included whether each image has a lesion\_id and whether the data is past data as auxiliary loss.  
      - backborn models  
          - convnext\_small.fb\_in22k\_ft\_in1k  
          - resnet18.fb\_swsl\_ig1b\_ft\_in1k  
          - swin\_small\_patch4\_window7\_224.ms\_in22k\_ft\_in1k  
          - resnet152.tv2\_in1k  
      - augmentation  
          - Resize(256 or 224), ShiftScaleRotate, VerticalFlip, HorizontalFlip, RandomBrightnessContrast, OneOf \[GridDistortion, OpticalDistortion\], Normalize, CoarseDropout  
      - optimizer  
          - optimizer : Adam  
          - lr\_head : 2e-4  
          - lr\_backbone : 2e-5  
      - scheduler  
          - CosineAnnealing with warmup  
      - batch\_size  
          - 32  
  - **Inference** 
      - randomly used 2 of the 5-fold models for each batch.  
      - LB 0.164, private 0.153 only image models average (local cv with past data 0.187 \~ 0.191)

### Tabular Models with image features

- Tabuler Model 3 
    - Feature Engineering  (Around 600 features)  
        - Based on public notebook  
    - Training with Past Metadata  
        - The objective is to address the issue of the small number of labels in the training data and the attribution that does not appear in the training data distributed for this competition.  
        - The validation set does not include past data.  
        - Tuning the \`scale\_pos\_weight\` parameter of LightGBM and XGBoost to deal with the difference between positive and negative ratios.  
    - Train with image oofs(4 models trained with past data) and use XGBoost and lightGBM.   
    - Create two patterns, excluding and excluding features that include attribution, and change the model used for inference.  
    - cv 0.183 public lb 0.175 private 0.165  
- Tabuler Model 2  
    - Feature Engineering (Around 3000 features by Polars):  
        - New Aggregation granularity: Add new aggregation of num\_cols by patient\_id, tbp\_lv\_location, attribution, tile\_type and their subsets.  
        - Shift Features: Aggregation of num\_cols by patient,  anatom\_site\_general , age\_approx, and shifts them from \-5(past) to \+5(future) units of age\_approxs within the same grouping, and diff of them (in real world we can't use future one...) 
        - Past Metadata Integration: Include information from patients who participated in both this and previous competitions.  
    - Soft Labels: indeterminate malignant records as target=1 and sample\_weight=0.5.  
    - Feature Selection: drop zero importance columns across all fold splits or not.   
    - Create two patterns, excluding and excluding features that include attribution, and change the model used for inference.  
    - train with image oofs(6 models)  and use XGBoost and lightGBM.   
        - single metadata only model \-\> cv: around 0.175 lb: 0.176, private: 0.165  
        - with img ensemble \-\>  **cv: 0.1837 lb: 0.181, private: 0.172**
- Tabuler Model 1
    - Feature Engineering (Around 300 features)  
        - grouping aggregation  
            - grouping  
                - patient\_id  
                - patient\_id \+ tbp\_lv\_location  
                - attribution  
                - attribution \+ tbp\_lv\_location  
            - aggregation  
                - mean  
                - std  
                - normalized feature; (numerical \- mean) / std  
  - model  
      - lgb  
      - learning rate 0.05, max\_depth 4  
      - no further parameter tuning  
      - no image pred feature input (weight average with image pred was better)  
  - Create two patterns, excluding and excluding features that include attribution, and change the model used for inference.
  - cv 171 public lb 177

### Ensemble

- Weighted average: optimizing the weights with Optuna to ensure their weights are as uniformly distributed as possible across all models.   
  (include both table with img output and img output only.)
- **cv: 0.1839 lb: 0.181, private: 0.172 (0.174 best…)**
