# 13th place solution

Competition: pii-detection-removal-from-educational-data
Rank: #13
Source: https://www.kaggle.com/c/pii-detection-removal-from-educational-data/discussion/497371

Thanks to Kaggle and the hosts for organizing this competition. I would also like to thank all the participants for posting discussions and providing useful datasets.

# Overview

My solution is based on an ensemble of NLP transformer models with different backbones, token lengths, and external datasets. I always run 4-fold experiments (split by `document % 4`) and evaluate the last checkpoints to find the best settings. For ensembling, I run the full-fit training with the same setting. All models are trained with focal loss.

# Modeling

My final ensemble consists of the following variations: 

Backbones:

- Deberta-v3-large
- Roberta-large

Token length and stride :

- 512 (128 stride)
- 1024 (256 stride)
- 2048 (256 stride)

External datasets:

- mpware’s dataset (https://www.kaggle.com/competitions/pii-detection-removal-from-educational-data/discussion/477989)
- nbroad’s dataset (https://www.kaggle.com/competitions/pii-detection-removal-from-educational-data/discussion/472221)

The external datasets were merged with the competition's data and used for training, without any downsampling.

The stability of the training varied across folds, and ultimately, I wanted to perform full-fit training (to get more diversity within the submission time), so I focused on finding settings that would be stable for all folds. All models were trained with focal loss, AdamW optimizer and cosine scheduler.

# Post-processing

- Setting a threshold for `O`. The value was optimized based on out-of-fold predictions.
- Setting thresholds for each class (slightly improved)

# Ensembling

The predictions of the spaCy tokens were taken from the first prediction of the corresponding token of each model, and the ensemble was simply averaged.

# Code
* Submission code: https://www.kaggle.com/yukiokumura1/pii-019-021-034-037-038-pp
* Github: https://github.com/okumura2997/kaggle-pii-solution
