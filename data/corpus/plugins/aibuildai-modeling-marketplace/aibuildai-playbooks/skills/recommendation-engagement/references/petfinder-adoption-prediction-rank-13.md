# 13-th place solution summary 0.44091 (65-th on public LB: 0.459~0.467)

Competition: petfinder-adoption-prediction
Rank: #13
Source: https://www.kaggle.com/c/petfinder-adoption-prediction/discussion/87733#latest-515056

First of all, thank you for organizers. It was my first time to seriously enter kernel competition, I enjoyed it.
I will write up my approach to summarize.

**[UPDATED] I published the kernel code**
 - [13-th place solution: ensemble of 5 models](https://www.kaggle.com/corochann/13-th-place-solution-ensemble-of-5-models)

## Feature Engineering
Many feature are adopted from [Single XGBoost model](https://www.kaggle.com/ranjoranjan/single-xgboost-model) as a baseline.
I will only write the additional feature engineering from this kernel.

### Tabular data
Applied cutoff for "Age", "Quantity", "VideoAmt" and "PhotoAmt".

Additional information I used is:
 - state gdp info (ref [The GDP and population of Malaysia states](https://www.kaggle.com/c/petfinder-adoption-prediction/discussion/78040)) 
 - breed ratings (ref [Cat and dog breeds parameters](https://www.kaggle.com/hocop1/cat-and-dog-breeds-parameters)): performance does not change so much. I adopted only 4 keys which are included both dogs &amp; cats.
 - language: 
English, Malay and Chinese are mainly used in Malaysia. Most of the description are English but some are Malay or Chinese. 
I detected language using `langdetect` library, and added these language as categorical feature.
Detect language is meaningful, because Malay people are mainly Muslim and do not have dogs (detail info can be found in [Stray animals in Malaysia: the Reality I Saw Travelling There For the Past Months](https://www.kaggle.com/c/petfinder-adoption-prediction/discussion/86581)).

### Text data
TFIDF --&gt; SVD feature extraction is executed for 3 text type 'Description', 'metadata_annots_top_desc', 'sentiment_entities' as same with Simple XGBoost model kernel.

Additionally, glove word embedding feature is extracted **only for "metadata_annots_top_desc"**.
When I tried to extract word embedding for description or sentiment_entities performance was worse.
I tried both [glove](https://www.kaggle.com/rtatman/glove-global-vectors-for-word-representation) and [fasttext](https://www.kaggle.com/facebook/fatsttext-common-crawl) word embedding and `glove.6B.200d.txt` was the best performance which I adopted as final submission.

### Image data
Final submission used `densenet` feature extraction (See [Extract Image features from pretrained NN](https://www.kaggle.com/christofhenkel/extract-image-features-from-pretrained-nn)).
Difference is that
 - I did not perform AveragePooling1D, and obtrain 1024 feature. I applied SVD to this 1024-dim vector.
 - More than 1 image may be contained in each pet, I calculated "mean" of image embedding (I calculated image embeddings up to 10 images per pet) feature when more than 1 image is contained.

Although I tested a lot of different models supported in ChainerCV (see [here](https://www.kaggle.com/c/petfinder-adoption-prediction/discussion/75943#486597) for supported models), I could not get better score than densenet and the final model used only the densenet. I was thinking Densenet performed better because it contains shallow &amp; deep information at once.

## Validation strategy
I applied GroupKFold with rescuerid, instead of StratifiedKFold done in many public kernels.
(Refer [this discussion](https://www.kaggle.com/c/petfinder-adoption-prediction/discussion/81809)).

When I check train and test data, it seems train and test rescuers are not overlapped.
GroupKFold shows much "worse" score in local experiment, but I think this is more proper setting.

I used 4-fold during fast experiment on local, and applied 10-fold for final submission to create 10 models for each architecture.

## Model
### XGBoost
It was quite fast to train models among these 5 models when using GPU, and I mainly tested my feature engineering only with XGB through this competition.
Since XGBoost cannot handle categorical data as opposed to LightGBM or CatBoost, I converted categorical values to numeric values by following
 - one hot encoding: Type, Gender, Vaccinated, Dewormed, Sterilized, State, FurLength, Health
 - breed encoding: Minor breeds are truncated as "unknown" and added breed1 &amp; breed2 as same field.
 - color encoding: Same with breeds, color1 color2 and color3 are added as same field.

### LightGBM
Since it can handle categorical feature, no additional feature engineering is performed.

### CatBoost
Same with LightGBM.

### xlearn
As written in these kernels ([The Hitchhiker's Guide to the PetFinder Competition](https://www.kaggle.com/c/petfinder-adoption-prediction/discussion/81597), [https://www.kaggle.com/c/petfinder-adoption-prediction/discussion/80937](https://www.kaggle.com/c/petfinder-adoption-prediction/discussion/80937)), using FFM for sparse category dataset seems to good to try. 
It was a problem that official kaggle kernel does not support `xlearn` library to use FFM, but @bminixhofer showed how to use it by adding library as a dataset, thank you! (Ref: [xlearn](https://www.kaggle.com/bminixhofer/xlearn))

It basically handles category data, so I made discretize to bin for the numeric data such as age &amp; fee. Other features (PhotoAmt etc) are simply used as categorical data.


### Neural Network (XDeepFM)
At first, I used simple MLP model but the performance is very bad.
During investigating `xlearn` approach, I found there are research which utilizes "FM" idea into neural network.
[xDeepFM](https://arxiv.org/abs/1803.05170) is one of the model which aims to apply many sparse categorical data. The model is also applied for [criteo dataset](https://www.kaggle.com/c/criteo-display-ad-challenge) in the paper. I implemented &amp; used this model.

Main problem for this competition is that model easily overfits.
Other than architecture, below Technics are used to reduce overfitting.

 - permutation augmentation: to reduce overfitting, I permute some column's value between rows for data augmentation purpose.
 - [spectral normalization](https://openreview.net/forum?id=B1QRgziT-)
 - dropout, weight decay
 - weight tying: same weights are used among the depth for CIN network inside xDeepFM.

I used most of the time for developing neural network, but sadly the performance was worst compared to the other models. However it is still effective to include for ensemble model, so it was not vain.

I used [optuna](https://github.com/pfnet/optuna) for hyper parameter tuning in the beginning-middle of the stage for neural network part. But I could not manage time to apply it for GBM &amp; xlearn models, so score might improve little bit more.


### Ensemble
Ensemble is performed by simply taking mean of each models.
Each model is trained with 10 models (using 10-folded GroupKFold), so final prediction is made by 10 * 5 = 50 models.

## Model performance summary

Below RMSE &amp; QWK are the value for validation data of `train` dataset, calculated by 10 GroupKFold on RescuerID.

| Model | RMSE | QWK (after optimized by nelder-mead) |
| --- | --- | --- |
| XGBoost | 1.041 | 0.448 |
| LightGBM | 1.047 | 0.439|
| CatBoost | 1.043 | 0.447|
| xlearn | 1.055 | 0.432 |
| NN(XDeepFM) | 1.065 | 0.414 |
| Ensembled | 1.038 | 0.457 |


## Final submission
How to determine threshold for regression model is quite important, and optimal way is not trivial in this competition (some [discussion](https://www.kaggle.com/c/petfinder-adoption-prediction/discussion/83870) can also be found).

I could consider following 4 options.

[Determine threshold]
1.Determine threshold to align with train histogram.
2.Determine threshold to align with histogram which gets good score on public LB data.

[Fix final prediction's histogram, instead of setting threshold.]
3.To align with train histogram.
4.To align with histogram which gets good score on public LB data.

I submitted A: 3 (0.459 on public LB) and B: 2 (0.467 on public LB) as final submission.

### [Updated] private leader board result
As I expected, threshold tuning for training histogram, method A, got high score = 0.44091 on private leader board which achieved me to reach gold medal in this competition.
method B got score 0.43866.


## Discussion
I have no idea how top kagglers got more than 0.470 scores and I am interested for those posts after the competition ends!

One interesting approach which I could not try during competition is to predict "ranking" rather than actual value. Since threshold is adjusted in post-processing, proper ranking is quite important to get high score in this competition.
