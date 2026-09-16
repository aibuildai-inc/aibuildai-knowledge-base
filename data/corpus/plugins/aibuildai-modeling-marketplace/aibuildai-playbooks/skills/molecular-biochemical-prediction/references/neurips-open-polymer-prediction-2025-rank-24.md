# 24th place solution - 2 stage Autogluon and XGB, no post-processing

Competition: neurips-open-polymer-prediction-2025
Rank: #24
Source: https://www.kaggle.com/c/neurips-open-polymer-prediction-2025/writeups/24th-place-solution-2-stage-autogluon-and-xgb

So I don't think I did anything particularly interesting, but since there was a big shakeout and I'm looking as which solutions survived it (beyond Tg postprocessing) I figured I might as well share mine.

My code consists of 20 different notebooks so I don't think I will share it. But I think it's pretty straightforward to understand.

##  Datasets

I used the datasets that were in most public notebooks, i.e. those added by the hosts to the competition data as well as the TgSS dataset. I didn't want to spend my time reading the fineprint and looking for hidden repositories.
I tried to process the datasets in 2 ways, first rescaling to be more similar to the train data, and then removing outliers. Both of those yielded inferior LB scores so I ended up using the raw data.
There were significant shifts between the datasets, for example to rescale the TgSS dataset I did *.71 + 22.

## Features

I generated basically all the features I could: Morgan, MACCS fingerprints, RDKit and Mordred descriptors, graph features, ChemBERT embeddings.
After dropping columns with low variance I ended up with around 1500 fields.

## Stage 1 models

First I ran a first autogluon with all the features. Then I used the feature importance functionality to keep around 500 per target and ran a second autogluon model. I also used the XGB model that was in several public notebooks such as [this one](https://www.kaggle.com/code/guanyuzhen/lb-0-064-neurlps-2025-baseline-random?scriptVersionId=250870055) (not sure if this the original author), although with my settings it performed significantly worse than the public version (.68 vs .64 LB).
As an example, the top features for Rg were 3 BERT generated fields, and Mordred bonding and structural information content.


## Stage 2 model

I used the 3 models from the first stage to generate pseudo labels on all the train SMILES. Then I trained a third Autogluon on these labels. I used all the first stage features as well as the predictions for other targets from the stage 1 models.

In general, the Autogluon ensembles consisted mostly of Tabm. For example the Tg model I ended up using was 77% TabM, 15% LGBM, 8% Catboost.

All Autogluon as well as well as feature selection were run on GPU notebooks with a 12 hours budget.

## Final solution

My .86 notebook (.66 on public LB) used the following models :

| Tg | 1st stage Autogluon with only 500 features |
|---|---|
| FFV | XGBoost |
| Tc| 2nd stage Autogluon (with weak labels)|
| Density| Average of both 1st stage Autogluon|
| Rg| 2nd stage Autogluon (with weak labels)|


The models were picked purely based on LB score. Although I didn't do any post-processing, the choice for Tg did bring a significant improvement, so it was probably the model with the "best" bias.

I submitted a more conservative ensemble with just the median of all the models I had tried, that one scored .65 on the public LB but only .91 on the private one.

## Rejected approaches

I originally just wanted to use this competition to try a bunch of tabular models and GNN. I tried the same 2 stage approach with LGBM, but it yielded inferior results. I also trained a GNN on Tg which performed catastrophically so I stopped there (in hindsight it maybe wasn't the best target).

In the end I think I was pretty lucky with my final score, I wasn't in bronze range on the public leaderboard. But seeing the other solutions I think it shows that Autogluon and TabM were pretty strong approaches.
