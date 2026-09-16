# 5th place solution [Updated]

Competition: lish-moa
Rank: #5
Source: https://www.kaggle.com/c/lish-moa/discussion/200533

Thank you to my team mate Anna, all participants, Kaggle Admin, Laboratory for Innovation Science at Harvard who realized this exciting competition. I would like to briefly share our 5th place solution.

## Problem settings
The most difficult problem in this competition seems to be that there are both seen and unseen drugs in the test set. This makes construction of the reliable validation difficult.

## Validation strategies and their properties
In this competition, there are mainly two types of validation strategies used. One is traditional MultilabelStratifiedKFold, and the other is MultilabelStratifiedGroupKFold which is a group awared one (aka Chris's method). Seeing predictions by models based on these schemes, MultilabelStratifiedKFold seems to give more confident predictions for seen drugs, whereas MultilabelStratifiedGroupKFold gives more balanced and robust predictions which are supposed to be suitable for unseen drugs. I thought that both are useful in different ways.


## Our approach
The setting where both seen and unseen labels are observed in the test set is similar to that in 
[Bengali.AI Handwritten Grapheme Classification](https://www.kaggle.com/c/bengaliai-cv19). So we decided to discriminate seen and unseen drugs by metric learning (L2 Softmax -> cosine similarity) and then blended models trained on each validation scheme based on the similarity. This separation of seen/unseen seems to be possible to some extent as shown in the figure below. This approach does not require something like pseudo-labelling & retraining. Simple forward processing is enough.


## Features
Raw cell and genes
PCA
rankGauss
PolynominalFeatures
Simple statistics like mean, min, max, skew, kurt, (X_train>9).sum(axis=1)
etc.

## Models
- ResNet-like shallow NN
- TabNet
For especially models for unseen drugs, we used simple networks.

## Reconstruction of drug_id [Updated]
From the very beginning of the competition, I was validating my models based on a method nearly equivalent to Chris CV.  For this purpose, I reconstructed drug_id by heuristic methods, including clustering and many manual adjustments. The official drug_id was published in the later stages of the competition, but many participants may have already overfitted. Once anyone has produced a good Public LB score, it's very hard to suspect that it's overfitting. The temptation and pleasure of running up LB is so intense.

Below are the CV results from both methods for the same model. My method worked almost as well as the Chris CV. I switched to Chris CV at the end of the competition because it performed a bit better. I think that Chris CV was good at handling drug_id with lots of observations and was similar to the actual splits.

| Method | LocalCV | Private | Public |
| --- | --- | --- | --- |
| reconstructed drug_id +my CV | 0.017173034 | 0.01613 | 0.01826 |
| official drug_id + Chris CV | 0.016907018 | 0.01612 | 0.01825 |

## Result
Public: 0.01812 (14th)
Private: 0.01602 (5th)
