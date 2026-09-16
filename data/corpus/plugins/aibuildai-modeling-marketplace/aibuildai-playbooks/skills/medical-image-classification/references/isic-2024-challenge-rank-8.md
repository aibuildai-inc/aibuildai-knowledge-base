# 8th Place solution (Trust your CV)

Competition: isic-2024-challenge
Rank: #8
Source: https://www.kaggle.com/c/isic-2024-challenge/discussion/532728

First, I would like to extend my congratulations to the winning team and express my gratitude to the Kaggle community and the competition organizers for another fantastic challenge, [ISIC-2024](https://www.kaggle.com/competitions/isic-2024-challenge/overview). This was my first serious attempt at  Kaggle medals competition and I learned a lot from it. Although I joined midway, I remained determined and persisted until the end.

One key lesson I learned was to trust my cross-validation (CV) score, even when it was lower than most public leaderboard (LB) scores. I focused on building a robust solution rather than being overly optimistic about high leaderboard rankings. Due to a hectic work schedule in August, I adopted a straightforward strategy: if an approach or idea didn't work on the first try, I would abandon it completely and move on.

## Data Preparation

My initial step was to thoroughly understand the dataset and follow key discussions in the forums to stay updated on what techniques were effective and what weren’t.

### Feature-engineering and selection.

For feature engineering, I started by building on top of public notebooks, adding one aggregate feature based on `tbp_lv_location`. Later in the process, I pruned features that had a `SAGE value of 0` as determined by my CatBoost model, which performed the best for me. Additionally, I removed four or five more features using a greedy selection approach — if removing a feature improved my CV score, I accepted the change.

### CV Selection
There was two options whether to choose startification or not but I dicieded to stick to the `StratifiedGroupKFold`   cv strategy though in my initial expermentation this didn't mattered much and used same cv throughout the competition so that it avoids me any kind of leakage due to differnet random seed in different stages of pipeline. This was my CV iterator for the entire contest.
```py
iterator = StratifiedGroupKFold(n_splits=5, random_state=6855, shuffle=True).split(
    train_meta["isic_id"],
    train_meta["target"],
    groups=train_meta["patient_id"],
)
```

### Modelling

My entire modelling could be dividied into four sections. 
	* Training GBDT solely based on tabular-data
	* Training Image model
	* Stacking the image on tabular-data to train GBDT.
	* Final ensembling all of them.

#### Stage-1 (Training GBDT on tabular data)
I had to do this stage twice as my initial approach was heavily influenced by public notebooks which resulted in unstable CV scores. Small changes would cause significant fluctuations in the leaderboard (LB) performance. Additionally, I was relying on early stopping, which negatively affected the CV-LB correlation.

In my second attempt, I managed to stabilize the CV score. Although it was slightly lower than in my first approach, the consistent pattern between CV and LB scores gave me confidence that I was moving in the right direction. With further tuning, I believed I could improve my score.

The models I focused on were **LGBM**, **XGBoost**, and **CatBoost**. While I had planned to try **IBM SnapML**, I ran out of time and resources to implement it. Here are a few key points from this stage:

- I didn't use any oversampling or undersampling techniques, although I had considered trying a bagging variation of them, but resource constraints prevented me.
- Instead, I used the inbuilt subsampling in XGBoost and CatBoost. For LGBM, the `pos_bagging_fraction` and `neg_bagging_fraction` worked so well that I didn't consider any kind of additional sampling technique.
- Since I avoided over and under resampling techniques, I found that using **scale_pos_weight** improved my overall score.

Summary table

| Model    | CV Score | Public LB | Private LB |
|----------|----------|-----------|------------|
| LGBM     | 0.17488  | 0.175     | 0.160      |
| CatBoost | 0.17330  | 0.173     | 0.159      |
| XGBoost  | 0.17508  | 0.175     | 0.157      |

**Note**: I employed a voting ensemble using three individual models ech for LGBM, CatBoost, and XGBoost. For each model, I selected the top three sets of hyperparameters. This ensemble approach resulted in a score improvement of 0.001.
### Stage 2: Training the Image Model

#### Pure Image Model

I had limited time for this stage as I spent most of it on Stage 1, leaving me with just about a week. Based on public discussions, I decided to use EfficientNet-B1, which was considered one of the best models. I experimented with EfficientNet-B0 but found that larger models like EfficientNet-B1 and B2 allowed me to train longer without the risk of overfitting or overly optimistic cross-validation (CV) scores. However, the downside to using larger models was the GPU memory limitations on Kaggle and the longer submission times, so I stuck with EfficientNet-B1 and did a small experiment with MaxVit-T.

For training, I followed this approach:

-   Instead of oversampling or undersampling, I used PyTorch's `WeightedRandomSampler`, assigning a weight of 1 to positive samples and 0.01 to negative samples.
    
-   I incorporated additional datasets from the 2018, 2019, and 2020 ISIC competitions. For these, I sampled all positive examples from each year and 10x the number of negative examples. In my `DataLoader`, I assigned the same weight to both positive and negative of extra at 0.01
    
-   Batch size had a significant impact, so I used a batch size of 64 for EfficientNet-B1 and 32 for MaxViT-T.
    
-   I used the `OneCycleLR` scheduler and observed that the number of iterations had a major influence on both CV and leaderboard scores. I trained EfficientNet-B1 for 200,000 iterations (i.e., 200,000 samples in the `WeightedRandomSampler`), and since MaxViT-T is a larger model, I extended its training to 300,000 iterations.
    
-   The following image transformations provided the best results for me:
    
    ```py
    train_transform = A.Compose(
    [
        A.RandomRotate90(),
        A.Flip(),
        A.ShiftScaleRotate(shift_limit=0.0, scale_limit=0.15, rotate_limit=90, p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.18, contrast_limit=0.12, p=0.5),
        A.HueSaturationValue(
            hue_shift_limit=3, sat_shift_limit=10, val_shift_limit=1, p=0.25
        ),
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.5),
        A.OneOf(
            [
                A.OpticalDistortion(distort_limit=0.5, shift_limit=0.0),
                A.ElasticTransform(alpha=0.2, sigma=6.0),
                A.GridDistortion(num_steps=2, distort_limit=0.2),
            ],
            p=0.7,
        ),
        A.Resize(224, 224),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ]
)
    
    val_transform = A.Compose([
        A.Resize(224, 224),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])
    ```
    

Below is a sample of my training setup similar to what @richolson pointed in this dicussion (https://www.kaggle.com/competitions/isic-2024-challenge/discussion/529457)
```py
pos_bagging_fraction = 1.0  # Weight for positives in WeightedRandomSampler
neg_bagging_fraction = 0.01 # Weight for negatives in WeightedRandomSampler
learning_rate = 1e-5
weight_decay = 0.0008
_scheduler = OneCycleLR(
    _optimizer,
    max_lr=0.0008,  # Peak learning rate
    steps_per_epoch=len(_train_loader),
    epochs=1,  # For one epoch
    pct_start=0.4,  # Spend 40% of time warming up
    anneal_strategy='cos'  # Cosine annealing
)
```

#### Image + Embedding Model

I saw significant improvement by incorporating tabular data into the pipeline alongside the image data. This approach allowed me to continue using additional datasets alongside the original dataset. I added an extra dimension to the input, indicating whether metadata was missing or not. For numerical features, missing values were filled with zeros. The 2020 ISIC dataset had a few metadata features similar to the 2024 ISIC data, which could have been useful, but due to resource constraints, I wasn’t able to explore that approach.

Following table summarizes my Image model experiment

| Model                    | CV-Score | Public LB | Private LB |
|---------------------------|----------|-----------|------------|
| Effnet-b1                 | 0.1546   | 0.150     | 0.148      |
| MaxVit-T                  | 0.157    | 0.151     | 0.144      |
| Effnet-b1 + MetaEmbedding | 0.170    | 0.162     | 0.154      |




## Stage 3 (Image Model + GBDT)

I was only able to use `Effnet-b1` oof predictions downstream didn't got enough time to put `Effnetb1 + MetaEmbedding` though my late take on it suggested that most gbdt models will have cv close to `~0.179x` using that.

Following table summarizes my models
In this stage, I divided the models into two parts: 
- **V1**: Trained on public features
- **V2**: Trained on pruned + additional features

| Model        | CV-Score | Public LB | Private LB |
|--------------|----------|-----------|------------|
| LGBM_V1      | 0.17606  | 0.178     | 0.168      |
| LGBM_V2      | 0.1766   | 0.176     | 0.167      |
| CatBoost_V1  | 0.1789** | 0.180     | 0.170      |
| CatBoost_V2  | 0.1780** | 0.180     | 0.171      |
| XGB_V1       | 0.1765   | 0.176     | 0.163      |
| XGB_V2       | 0.1768   | 0.177     | 0.166      |

**Note**: The CatBoost model scores fluctuated between [0.1792 - 0.1781] due to GPU training variability.


## Stage 4 (Ensembling)

I did it on last day using last 5 submissions and couldn't think of better strategy than oputna voting and the ending result had same perfomance as my catboost model :(. 

Here's my pipeline. For optuna ensembling I used `CmaEsSampler`to find the weights for averaging.



The final score was as follows 

**CV**: ~0.1790 (This also fluctated due to variation in Cmaes Sampler and Catboost GPU)
**Public LB**: 0.180
**Private LB**: 0.171

Here are links to my notebooks
[Best-Image-model](https://www.kaggle.com/code/uryednap/efficientnet-isic-2024-eval/notebook)
[Best-Meta-Image-model](https://www.kaggle.com/uryednap/efficientnet-meta-isic-2024-eval)
[Final-ensemble](https://www.kaggle.com/code/uryednap/final-approach-isic-2024/notebook)


## Potential Improvements (Planned but Not Tested)

I believe there are additional strategies that could have potentially boosted the score. Unfortunately, due to time and resource constraints, I couldn't explore them fully. I’m unsure if these approaches would work, but I’d love to hear from anyone who has tried them:

1. **Using MetaEmbedding Image Model OOF-Preds Downstream**: Incorporating out-of-fold (OOF) predictions from the MetaEmbedding Image model into downstream models might have improved the results.
2. **Using Last-Layer CNN Features for GBDT**: Extracting features from the last layer of a CNN model and using low-dimensional projections (PCA, SVD, or other techniques) as inputs for GBDT models could have provided better feature representation.
3. **Combining Transformers or Factorization Machines with Image Models**: My initial attempt at integrating TabTransformer with image models wasn’t successful. However, combining transformers or factorization machines with image models might work with sufficient compute power and large batch sizes.
4. **Fitting Tiny Image Models on Random Forest Leaves**: I had a somewhat unconventional idea of fitting tiny versions of image models to the leaves of a random forest and somehow combining the predictions. I didn’t try this, as I wasn’t sure if it was a viable approach or worth the effort.
5. **Third Stage Stacking**: I had a local run using catboost as meta-estimator for stacking third stage predictions but wasn't able to explore more as I had exhausted my final 5 submissions though it yielded a better validation score than the weighted average.


## What worked.
1. Using aggregation features especially `patient_id` and `tbp_lv_location`.
2. Using the `OneCycleLR` technique instead of standard epoch-wise training (I was initially doing 2 epochs)
3. Using WeightedRandomSampler for pytorch Image training. (I was doing manual global down-sampling)
4. Not using Early-Stopping (prevented me to be overoptmistic about my result).
5. Incorporating additional dataset from past ISIC competition.
6. Using Metadata for improving the score of Image model.


## What Didn't Work (Tried Only Once)

1. **LGBM Linear Variation**: The CV-LB pattern was unstable, and the presence of missing values in the test set led to numerous failed submissions and didn't tried for stage-3.
2. **Transformers (TabTransformer) with Image Model**: This approach didn't yield the expected results when combined with the image model.
3. **Two-Stage Tuning for Image Model**: The strategy of fine-tuning on the combined dataset (original + extra) in the first stage and then fine-tuning again with meta-embeddings and the stage-one image model did not produce meaningful improvements.
