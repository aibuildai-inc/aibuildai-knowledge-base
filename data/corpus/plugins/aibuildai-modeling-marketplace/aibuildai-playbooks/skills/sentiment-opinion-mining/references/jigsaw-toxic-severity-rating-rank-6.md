# 6th Place solution

Competition: jigsaw-toxic-severity-rating
Rank: #6
Source: https://www.kaggle.com/c/jigsaw-toxic-severity-rating/discussion/306926

## Overview

Congradulations, all winners.  
But really lucky for us, We got a gold medal. None of our  members could have imagined this result. As pointed out [in 7th place solution](https://www.kaggle.com/c/jigsaw-toxic-severity-rating/discussion/306366), we had a lot of models that were trained mainly on Jigsaw 1st directly, and it is possible that the performance was better for private tests. This was simply luck.


## Data
- [jigsaw-toxic-comment-classification-challenge(jigsaw_1)](https://www.kaggle.com/julian3833/jigsaw-toxic-comment-classification-challenge)
- [ruddit jigsaw dataset](https://www.kaggle.com/rajkumarl/ruddit-jigsaw-dataset)
- [jigsaw_regression_data](https://www.kaggle.com/nkitgupta/jigsaw-regression-based-data)

When we used jigsaw_1 to train a regression models, we calculated target values by following  formula.

 ```(y = toxic0.32 + severe toxic1.82 + obscene0.16 + threat1.5 + insult0.64 + identity_hate1.5) ```


## CV strategy
We used validation_data.csv to evaluate cv score. (In order to evaluate more accurately,) We removed all sentences from the training data that overlapped with validation_data.

## Preprocessing / Engineering
### <I>mogmog part</I>
The training data were jigsaw1(applied under sampling) and Ruddit. We apply multiple patterns of text preprocessing. After preprocessing,  TF-IDF (and SVD for lgb) , gensim_embedding, Basic feature(Word count, Character count), FastText embedding were applied on each of them to generate features.

### <I>kma### part</I>
The training data were jigsaw_1, ruddit, jigsaw_regression_data. we applied text cleaning on jigsaw_1 and ruddit, and none on jigsaw_regression_data.
we applied TF-IDF for each of them, and trained Ridge reg. (alpha=0.5, 1, 2).
In inference, alpha=0.5, 1, and 2 were used as an ensemble.

### <I>kfsky part</I> 
The training data was jigsaw1 data (applied under sampling, text cleaning) for training data and generate features.
- TF-IDF(and SVD)
- CountVectorizer(and SVD)
- Word count
- Character count
- FastText（100dim）

Train LightGBM with these features.

And we used TF-IDF and trained Ridge model.

## Training Model

we trained following models.

|Model|Dataset|Text Cleaning|Feature|CV|
|---|---|---|---|---|
|LightGBM_kfksy_1|jigsaw_1|Cleaned|TF-IDF, CountVectriser, FastText, Word count, Character count|0.6843|
|Ridge_kfksy_1|jigsaw_1|Cleaned|TF-IDF|0.6639|
|Ridge_kma###_1|jigsaw_1|Cleaned|TF-IDF|0.6823|
|Ridge_kma###_2|ruddit|Cleaned|TF-IDF|0.6312|
|Ridge_kma###_3|jigsaw_regression_data|Non|TF-IDF|0.6713|
|Ridge_mogmog_1|jigsaw_1|Cleaned|TF-IDF|0.6810|
|Ridge_mogmog_2|ruddit|Cleaned|TF-IDF|0.6283|
|lightGBM_mogmog_1|jigsaw_1|Cleaned|gensim_Embedding & FastTextEmbedding & Basic_Feat|0.6792|
|lightGBM_mogmog_2|ruddit|Cleaned|gensim_Embedding & FastTextEmbedding & Basic_Feat|0.6572|
|lightGBM_mogmog_3|jigsaw_1|Cleaned|TF-IDF & SVD|0.6711|
|lightGBM_mogmog_4|ruddit|Cleaned|TF-IDF & SVD|0.6130|
|RoBERTa_base_mogmog_1|jigsaw_1|Cleaned|Only text cleaning|0.6879|


## Weight Ensembling
We ensembled these models. Before ensemble, we applied MinMaxScaler because the scale of the prediction values were different among the models.  The weights of each model are determined so that the CV score is maximized. (we used ```scipy.optimize.minimize(method='powell')```) .  After ensembling, cv score was 0.7048.

## Closing
Thank you [mogmog](https://www.kaggle.com/ryo1993), [kma###](https://www.kaggle.com/kmakmar), [mkt0309](https://www.kaggle.com/mkt0309) for teaming up with us.

## Clean Inference Code
https://www.kaggle.com/ryo1993/inference-krkm-final-001?scriptVersionId=88284870
