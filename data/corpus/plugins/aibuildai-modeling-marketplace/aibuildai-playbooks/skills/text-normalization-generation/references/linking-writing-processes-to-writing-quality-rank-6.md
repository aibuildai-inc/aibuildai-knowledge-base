# [CPU Only]６th Place Solution for "Linking Writing Processes to Writing Quality"

Competition: linking-writing-processes-to-writing-quality
Rank: #6
Source: https://www.kaggle.com/c/linking-writing-processes-to-writing-quality/discussion/467848

[Pipeline diagram]

## Trust CV is important
Based on the following assumptions, I hypothesized that notebooks with better CV (Cross-Validation) scores, rather than LB (Public Leaderboard) scores, are likely to perform well on the Private LB.

1. From the following observations, I inferred that public notebooks with good LB scores were overfitting to the LB:

    - There was a correlation between my CV scores and LB score.
    - When I calculated the CV scores of public notebooks that had better LB scores than mine, I found that my notebooks had better CV scores.
    - Adding some features I created to the public notebooks improved both CV and LB scores.
2. From the following observations, I speculated that the LB contains more instance of easier to predict data than the training data. Additionally, I surmised that fitting to the LB can lead to a less robust model and increase the likelihood of shake-down on the Private LB:

    - In experiments, all notebooks had better LB scores than CV scores.


<br>

## Context
- Business context: https://www.kaggle.com/competitions/linking-writing-processes-to-writing-quality/overview
- Data context: https://www.kaggle.com/competitions/linking-writing-processes-to-writing-quality/data

---

## Overview of the approach

I performed all the computations on the CPU, without utilizing the GPU.

<br>

### Preprocessing

I refer to the [essay constructor notebook](https://www.kaggle.com/kawaiicoderuwu) by [@kawaiicoderuwu](https://www.kaggle.com/kawaiicoderuwu) to reconstruct the essays.

<br>

### Feature Engineering & Feature Selection

All the features I generated, along with the reconstructed essays, have been made available as a [public dataset](https://www.kaggle.com/datasets/kazukiigeta/linking-feature-engineering-data001/data).

In terms of feature engineering, my approach aimed to keep the number of features as low as possible, while adding features that I thought were really effective. The total number of features were only 169.
This was because the sample size was quite small, 2471, and I felt that having several hundreds of features was too many.

I referred to the [silver bullet notebook](https://www.kaggle.com/code/awqatak/silver-bullet-single-model-165-features) created by [@awqatak](https://www.kaggle.com/awqatak) in my Kaggle solution. 

For instance, I conducted Feature engineering & selection like the following:
 
- Following the acquisition of character-level TF-IDF vectors for the reconstructed essays, I introduced additional features, such as incorporating a vector created by compressing the TF-IDF vectors with an n-gram range of (1, 5) into 29 dimensions using TruncatedSVD.
- Introducing a feature to express the number of times the 'Ctrl' key was pressed (as I believed it could be relevant to the quality of the essay).
- Removing features that exhibited a correlation coefficient of 1 or close to 1 with each other, among other modifications after feature engineering.

<br>

### Validation

I used Repeated Stratified K-Fold Cross Validation to enhance the reliability of the OOF predictions and CV scores.

```
for i in range(n_repeats):
    skf = StratifiedKFold(n_splits=5, random_state=seed ** i, shuffle=True)
```

<br>

### Models/algorithms

I decided to prioritize diversity in model selection rather than increasing the number of features in my solution. To achieve this, I opted for various types of models within my approach.

For many of these models, I ensured diversity in feature combinations through feature bagging.

Additionally, I implemented Random seed averaging to further enhance the performance.

<br>

#### Level-0 model

- LightGBM Classifier
- LightGBM Classifier (ExtraTrees)
- LightGBM Regressor
- LightGBM Regressor (ExtraTrees)
- CatBoost Regressor
- XGBoost Regressor
- RandomForest Regressor
- Ridge
- Lasso
- KNN
- SVR
- LightAutoML
	- MLP
	- Dense Light
	- Dense
	- ResNet
- TabPFN
- TabNet

#### Level-1 model

- ExtraTree + Bagging
- BayesianRidge
- MLP

#### Level-2 model
- Geometric mean blending

<br>

### Postprocessing
  
It's possible that the test data for the Private LB includes essays with a score of 0, which aren't present in the training data. However, I believed that typing behavior alone is insufficient to accurately predict essays that would score 0, and unanonymized essays would be necessary for this task. Therefore, I set the lower limit for clipping to 0.5.

```
np.clip(submission, 0.5, 6.0)
```

<br>

### What didn't work
- Word2Vec for the reconstructed essays
- Word-level TF-IDF
- NN models for the raw time-series data
