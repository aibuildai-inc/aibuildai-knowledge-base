# 9th place solution - DINOv2 + CatBoost = 🚀

Competition: planttraits2024
Rank: #9
Source: https://www.kaggle.com/c/planttraits2024/discussion/510188

## Acknowledgements 
First of all, I would like to appreciate the work done by the competition hosts and other competitors, this journey was quite interesting and fun! I think it was the first time I really had to work with multimodal data so I learned a lot during this competition. 

## Approach #1 - train one big model
I started experimenting with [this notebook](https://www.kaggle.com/code/markwijkhuizen/planttraits2024-eda-training-pub), trying to leverage different losses, backbones and other hyperparameters. This approach was promising, some things were working, however I quickly ran out of resources - the model was training too long, the batch size was too small, so I switched to the new tactics. 

## Approach #2 - embeddings + boosting
I discovered [this discussion](https://www.kaggle.com/competitions/planttraits2024/discussion/487179) (and other posts by the same person) and thought it was a good idea - to extract embeddings from some big open-source image model, concat them with tabular data and pass them to boosting. 
For the boosting I tried XGBoost and CatBoost and CatBoost was doing much better, especially when I discovered that it can work directly with embeddings (it can treat them not only as numerical features, but also do some dimensionality reduction on them), it helped a lot. 
As for the embeddings part, I tried quite a few models - from CLIP-like (EVA-CLIP, MetaCLIP etc) to huge multimodal models like ImageBing or InternVL, but for some reason one model was ahead of the others - it was DINOv2 [giant]. So I tuned some hyperparameters of CatBoost and pretty much got my best result. 

## Approach #2.1 - polynomial features, blending and other Kaggle tricks
At this time I wasn't really hoping to gain much more R2 score and I tried some tricks. I don't remember them all but there were 2 insights for sure: 1) blending different models trained on different embeddings didn't work; 2) adding some number of 2-degree polynomial features helped a little. 

## Result
I've published [my final notebook](https://www.kaggle.com/code/korotas/9th-place-planttraits2024-dinov2-catboost) with **public score 0.51162** and **private score 0.51238**, also with some additional comments and more approaches that I've tried, I hope it can be useful to someone.
