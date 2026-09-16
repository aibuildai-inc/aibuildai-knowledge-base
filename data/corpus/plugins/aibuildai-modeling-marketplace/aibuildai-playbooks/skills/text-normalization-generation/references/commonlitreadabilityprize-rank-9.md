# 9th place solution -Ensemble of four models

Competition: commonlitreadabilityprize
Rank: #9
Source: https://www.kaggle.com/c/commonlitreadabilityprize/discussion/259982

To begin with, We would like to thank Kaggle and Commonlit for hosting such an interesting competition! This is our first challenge in the field of NLP competition and very thanks to @rhtsingh, @maunish and @andretugan for sharing their wonderful kernels.  Their notebooks were great help to making this competition a great one for sure.

## Summary
We chose the one with best Public LB and the one with the best CV as our final submissions. The following are our best submissions.

|   | Public LB |Private LB|
| --- | --- | --- |
| best Public LB | 0.450 | 0.451 |
| best CV | 0.451 | 0.448 |

We ensembled 10 models including two 25-fold CV models.

## Cross validation strategy
The dataset size is one of the most important factor for machine learning. In this competition, train.csv contained about 3000 samples. Generally speaking, this number is very small for training deep learning models. To exhaustively utilize this small dataset, we carefully determined the number of folds (K) in K-fold cross validation. Our hypothesis was that if we increased K, models would perform better thanks to larger number of training samples (Of course, there would be an upper limit of K because the number of validation samples becomes too small). We tried 4 candidates(K=5,10,15,25) to test this hypothesis. As we expected, we observed small but stable improvement of LB score with larger K. To be more specific, increasing K from 5 to 15 improved public (and private) LB by 0.002. Increasing 15 to 25 did not change the displayed score but improved the score slightly (we confirmed that by sorting the submissions by the scores).We decided to employ 25 folds for some strong single models (deberta-large, roberta-large trained by cross entropy loss), paying attention to inference time. As a side note, we used normal k fold cross validation because stratified k fold using binned target did not seem to improve cv-lb correlation much. 

## Models
Like most teams, we used pretrained models from huggingface implemented in pytorch. Through several experiments,and found that deberta-large performed significantly better. However,deberta-large required much longer training time than other models, so we attempted to obtain the optimal parameters for deberta-large in a short time by using deberta base. Several kernels and papers were very helpful in this experiments (e.g.: https://arxiv.org/abs/2006.04884) . After changing many parameters, including lerning rate, activation layer, the best results were obtained by changing loss function. Some models using cross-entropy loss have shown excellent results in previous NLP competitions(e.g. :https://www.kaggle.com/c/google-quest-challenge/discussion/129978). So we trained BCEWithLogitsLoss using Sigmoid(target-Median(target)) as the prediction label. Median(target) was added to the regressor during prediction.This idea significantly contributed to both CV and LB score, and also contributed significantly to the ensemble. 

## Weights
|Model  | Public | Weight| 
| --- | --- | --- |
|  RoBERTa-Large (25 fold cross entropy)| - |0.4 |
|DeBERTa large (25fold)|	0.458	|0.25|
|DeBERTa large (15 fold cross entropy)|-|	0.25|
|DeBERTa large 5epochs 5fold |0.460	|0.1|

##  WHAT did not work
* large model
    * deberta-xlarge
* other model
    * ernie_large
    * xlnet-base-cased
    * albert-base-v1
    * distilbert-base-uncased
    * xlm-roberta-base
    * microsoft/deberta-base
    * nghuyong/ernie-2.0-en
    * nghuyong/ernie-2.0-large-en
* CV strategy
    * stratified k-fold by target
    * stratified k-fold by std
* pseudo labeling
* data argumentation
* post processing
* SWA

## Computing environments
* google colab pro *3 
* AWS p3.8xlarge (for deberta-xlarge, not worked)
* GCP v100*4 (for deberta-xlarge, not worked)
