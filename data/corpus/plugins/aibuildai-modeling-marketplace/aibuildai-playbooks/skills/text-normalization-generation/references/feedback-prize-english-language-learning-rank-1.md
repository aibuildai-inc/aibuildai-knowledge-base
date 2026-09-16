# 1st Place Solution

Competition: feedback-prize-english-language-learning
Rank: #1
Source: https://www.kaggle.com/c/feedback-prize-english-language-learning/discussion/369457

Thanks a lot to the hosts and Kaggle for hosting this interesting competition, we had great fun working on this one. Also congratulations to all other competitors for the great solutions and results. Special thanks to my teammates @evgeniimaslov2 and @poteman for such a perfect teamwork. 

## Summary: 
Our solution is based on training lots of different models with different pooling techniques and max lengths and finally ensembling. We also used 3 models to extract embeddings and add them to ensemble. 

## Cross validation

Throughout this competition we had near perfect correlation between ensemble CV and LB. Whenever we saw some improvement on CV, we saw it reflected in a similar manner on the LB with very small random range. For splitting the folds, we just used MultilabelStratifiedKFold startegy. 

## Modeling: 
Our final solution is a combination of different modeling approaches. 

Train/PL data + Different Pooling techniques + model + max_len(768/1462 for deberta models, 512 for others) + freezing top n layers + re_init/no re_init  top layer + differential LR + AWP. 

### Poolings Used: 
1. MeanPooling
2. ConcatPooling
3. WeightedLayerPooling
4. GemPooling
5. LSTMPooling

### Models Used: 
1. microsoft-deberta-v3-base
2. deberta-v3-large
3. deberta-v2-xlarge
4. roberta-large
5. distilbert-base-uncased

Model Used for Embeddings and SVR
1. "facebook/bart-large"
2. "flax-sentence-embeddings/all_datasets_v3_roberta-large"
3. "facebook/bart-large-mnli"

Public Models: 


### Weight Tuning: 

* We used optuna to tune the model weights, since we used different folds for training, we used final oofs to tune the weights. Also weights are tuned target wise.
* We only add a model to our ensemble if it improves both our Ensemble CV/LB.
* The best CV we get using this approach was 0.44073 (We didn't select this sub fearing overfitting but this has best private score.) 
Our Second Best CV was 0.44096 (It has good CV/LB correlation).
Model | CV | Public LB | Private LB |
| --- | --- |--- |--- 
| deberta-v3-base with PL (Roh) | 0.4464 | 0.437219 | 0.437982
| deberta-v3-large (Yev) | 0.4460 | 0.436758 | 0.434625
| deberta-v3-large (pub) |0.4548 | 0.439502 | 0.437965
| nischay (pub) |4588| 0.442645 | 0.439982
| deberta-v2-xlarge (pub) | 0.4675 | 0.442497 | 0.443604
| roberta-large (Roh) |0.4596 | 0.443616 | 0.444081
| deberta-v3-large (pub) | 0.4552 | 0.440045 | 0.437234
| deberta-v3-large (Roh) | 0.4556 | 0.444469 | 0.440331
| distilbert-base-uncased (Roh) | 0.4794| 0.458219 | 0.459658
| deberta-v3-large (yev) | 0.4440 |  0.435268 | 0.434334
| deberta-v3-base with AVG(trainPl + train) (Roh) | 0.4482 | 0.43856 | 0.441147
| roberta-large with AVG(trainPl + train) finetune train (Roh) | 0.4649 | 0.449538 | 0.447643
| deberta-v3-large (yev) | 0.4538 | 0.441582 | 0.439262
| deberta-v3-large (yev) | 0.4617 | 0.448342 | 0.442106
| deberta-v3-large (yev) | 0.4447 | 0.436268 | 0.435978
| fb_bart_large_svr  (Roh) | 0.4635 | 0.455645 | 0.453132
| ad_v3_roberta_large_svr (Roh) | 0.5113 | 0.501901 | 0.500084
| fb_bart_large_mnli_svr (Roh) | 0.4708 | 0.456652 | 0.459615
|Final Ensemble CV | 0.44096 | 0.433821 | 0.433356

We also noticed that generating Soft/Pseudo labels on train data leads to huge overfitting, CV around 0.437. 

## Pseudo labels: 
1. FB1 data
2. Train data
3. Mean of Train Pl / Actual Labels  [This gives nice boost in both CV/LB]


## What Worked: 
1. Different Pooling techniques
2. Different max_len
3. freezing top n layers
4. re_init top n layers.
5. Training with Differential learning rate
6. Pseudo labels

## What didn't worked
1. Augmentations
2. Post processing

## Important Citations:
https://www.kaggle.com/code/cdeotte/rapids-svr-cv-0-450-lb-0-44x
https://www.kaggle.com/code/nischaydnk/fb3-pytorch-lightning-training-baseline
https://www.kaggle.com/code/yasufuminakama/fb3-deberta-v3-base-baseline-train
https://www.kaggle.com/code/kojimar/fb3-deberta-family-inference-9-28-updated
https://github.com/ybabakhin/kaggle-feedback-effectiveness-1st-place-solution

## Extra Links: 
Link to our best submission  https://www.kaggle.com/code/rohitsingh9990/merged-submission-01/notebook?scriptVersionId=111953356

## Thanks and Acknowledgements:
Special thanks to @cdeotte, @yasufuminakama, @nischay, @nischaydnk, @e0xextazy and @kojimar for sharing their notebooks/datasets/ideas publicly. 

## Team Members
@rohitsingh9990
@evgeniimaslov2
@poteman

Training code can be accessed from https://github.com/rohitsingh02/kaggle-feedback-english-language-learning-1st-place-solution
