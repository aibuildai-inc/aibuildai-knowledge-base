# [9th Place] Model Predictions as Features

Competition: playground-series-s3e10
Rank: #9
Source: https://www.kaggle.com/c/playground-series-s3e10/discussion/396374

Thanks to Kaggle for another great competition and especially to all those who participated and shared models.

My approach here was broadly similar to that in the Gemstone Price competition, in that I used model predictions as additional features. However, I took a more conventional approach to generating predictions for the training set, which came simply from a cross-validation. Specifically, I took the following models:

Model 1: [PS S3E10, 2023 EDA and Submission](https://www.kaggle.com/code/sergiosaharovskiy/ps-s3e10-2023-eda-and-submission) by @sergiosaharovskiy v15.

Model 2: [PS S3E10 | Pulsar research | 0.0321](https://www.kaggle.com/code/dmitryuarov/ps-s3e10-pulsar-research-0-0321) by @dmitryuarov v1.

Model 3: [PS3E10 EDA| XGB/LGBM/CAT Ensemble Score 0.03174](https://www.kaggle.com/code/tetsutani/ps3e10-eda-xgb-lgbm-cat-ensemble-score-0-03174) by @tetsutani v29.

Model 4: [PS s3e10 FLIM-FLAML thank you ma'am](https://www.kaggle.com/code/paddykb/ps-s3e10-flim-flaml-thank-you-ma-am) by  @paddykb v1.

Model 5: [PS S3 E10](https://www.kaggle.com/code/alexandershumilin/ps-s3-e10) by @alexandershumilin v5.

From each of these, I generated predictions for the training (via cross-validation), test and original datasets. The predictions of Model 1  are then treated as a new feature, as are each of those of Models 2-5. This provides an additional five features that I subsequently use for prediction.

The five models were rerun using the expanded feature sets to generate what I call Models 1M, 2M, 3M & 4M (there was no 5M, as this turned out identical to the predictions of Model 5). 

I combined these models with an additional submission file (Model 1A) from Model 1, and these further five public models:

Model 0: [PS s3e10 GAM - Finger on the pulsaRRRRR](https://www.kaggle.com/code/paddykb/ps-s3e10-gam-finger-on-the-pulsarrrrr) by @paddykb v2.

Model 6: [PS3E10: R-GAM](https://www.kaggle.com/code/syerramilli/ps3e10-r-gam) by @syerramilli v4.

Model 7: [https://www.kaggle.com/code/eamonntweedy/playground-s3e10-pulsars-gbdt-ensemble-optuna](https://www.kaggle.com/code/eamonntweedy/playground-s3e10-pulsars-gbdt-ensemble-optuna) by @eamonntweedy v1.

Model 8: [PS3E10 : Ensemble Model Score - 0.03138](https://www.kaggle.com/code/ashenranaweera/ps3e10-ensemble-model-score-0-03138) by @ashenranaweera v5. 

Model 9: [PG3E10 | MLJAR](https://www.kaggle.com/code/andreychubin/pg3e10-mljar) by @andreychubin v4.

Thus, if we keep count, there are now 15 models to ensemble: 0, 1, 1A, 1M, 2, 2M, 3, 3M, 4, 4M, 5, 6, 7, 8, 9.

I tried this in two ways. In the Gemstone competition a simple median did well, but here I found that [a very loosely fitted Boltzmann ensemble](https://www.kaggle.com/jbomitchell/pulsar-boltzmann-ensembler) was better - this was parameterised to have model weights totalling the equivalent of about six models.
