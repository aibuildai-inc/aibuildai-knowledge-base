# 4th Place Solution

Competition: lish-moa
Rank: #4
Source: https://www.kaggle.com/c/lish-moa/discussion/200808

First of all,  We([@Kanna Hashimoto](https://www.kaggle.com/kannahashimoto) , [@e-mon](https://www.kaggle.com/eemonn), [@Konbuiyon](https://www.kaggle.com/kento1993), [@hatry](https://www.kaggle.com/toseihatori), [@hikomimo](https://www.kaggle.com/hikomimo)) would like to thank you to Kaggle and the hosts for hosting the competition.
I'm still excited about this result!


## Solution Overview



Our approach is very simple. Trust CV & Ensemble.

## All of the source code is here

- [github](https://github.com/e-mon/lish-moa)

- [kaggle notebook](https://www.kaggle.com/kento1993/nn-svm-tabnet-xgb-with-pca-cnn-stacking-without-pp)

## Validation Scheme
@cdeotte 's CV methods (Drug and MultiLabel Stratification) w/o cp_type = 'ctl_vehcle'.
This method could almost completely correlate CV with LB.

(Thx @cdeotte !!)

[Drug and MultiLabel Stratification Code](https://www.kaggle.com/c/lish-moa/discussion/195195)

## Feature Engineering

- statistical features
    - sum, mean, std, kurt, skew, median
- PCA
    - applied only CELL features
- combination features
    - difference between the two features
    - Because of combinatorial explosion (872C2=379_756), we used only features with variance above the threshold.

     ```
    for c in itertools.combinations(features_g + features_c, 2)):
        col_name = f"{c[0]}_{c[1]}_diff"
        d = train_features_df[c[0]] - train_features_df[c[1]]
        diff_val = np.var(d)
        if diff_val > 15:
            train_features_df[col_name] = d
            var_list.append(diff_val)
    ```

- quantile transformer
    - except for PCA features

## Modeling

#### 1st stage models
- stage1-NN (7folds 5seeds)
    - stacking of 2 models 
        - stage0-NN1 (7folds 5seeds)
            - labels : scored + nonscored targets (except for all 0 columns)
        - stage0-NN2 (7folds 5seeds)
            - labels : scored targets
- stage1-tabnet (7folds 5seeds)
    - labels : scored targets
- stage1-svm (4folds 1seeds)
    - labels : scored targets
    - create 206 models
- stage1-xgb (5folds 2seeds)
    - labels : scored targets
    - create 206 models

|    | model             |   cv score | blend weight|
|---:|:------------------|-----------:|------------:|
|  0 | stage0-NN2        |   0.016779  |            -|
|  1 | stage1-Tabnet     |   0.016846  |        0.228|
|  2 | stage1-SVM        |   0.017554  |        0.160|
|  3 | stage1-NN         |   0.016736  |        0.598|
|  4 | stage1-xgb        |   0.017366 |       0.0123|

- correlation matrix for 1st stage models
.png?generation=1606833799313456&alt=media)

#### 2nd stage models

- 2D-CNN stacking (5folds 5seeds)
    - model-wise CNN for 4 stege1 models 
    - [iMet 7th place solution](https://speakerdeck.com/phalanx/imet-7th-place-solution-and-my-approach-to-image-data-competition?slide=24)
    
- weight optimization
    - Scipy minimize function ('Nelder-Mead')
    - [TReNDS 2nd place solution](https://www.kaggle.com/c/trends-assessment-prediction/discussion/162765)

|    | model                  |   cv score |
|---:|:-----------------------|-----------:|
|  0 | 2D-CNN stacking        |   0.01670  |
|  1 | weight optimization    |**0.016518** |

#### postprocessing
- Drug ID Prediction
    - Multi-class prediction with LightGBM for frequently appeared 8 drug id's in train datasets.
    - replacing the label of a record whose prediction exceeds the threshold with the hard target of its drug_id
    - target labels
        - 9 classes: ['87d714366', '9f80f3f77', '8b87a7a83', '5628cb3ee', 'd08af5d4b', '292ab2c28', 'd50f18348', 'd1b47f29d', other_drug_ids]
    - metric
        - accuracy: 0.984965 with 5folds 15seed averaging


#### final submission

Our final two submissions are averaging of two predictions (2D-CNN stcking + weight optimization averaging with 1st models ) w/ and w/o postprocessing.

- avg weighted_blend + CNN (CV): 0.01647 (PublicLB: 0.01816/PrivateLB: **0.01600**)
- w/ postprocessing  (CV): 0.01644 (PublicLB: 0.01811/PrivateLB: **0.01618**)
    - [notebook w/ postprocessing](https://www.kaggle.com/eemonn/nn-svm-tabnet-xgb-with-pca-cnn-stacking?scriptVersionId=48144329)

Our training & inference processes were almost done only on Kaggle  kernel.
