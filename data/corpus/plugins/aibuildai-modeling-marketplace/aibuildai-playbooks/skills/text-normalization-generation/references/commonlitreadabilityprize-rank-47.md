# [Private 47th / Public 55th] Stable local cv & public leaderboard using stacking with linear models and forward selection

Competition: commonlitreadabilityprize
Rank: #47
Source: https://www.kaggle.com/c/commonlitreadabilityprize/discussion/258594

In this competition, our team "3 of a kind" (@steubk @pietromarinelli @lucamassaron) used a stacking strategy, trying to mainten our solution as simple as possible (given the circumstances of the competition).

1. we used a stratified cv 5-fold strategy, based on the target distribution which has been divided into 25 parts using pd.cut. Our local cv always followed the LB improvements and we didn't have any doubt in the end if to choose the LB or the local cv. We decided for our two final submissions based on their diversity. We always recorded the OOF for every model and stored each fold's model.

2. We experimented different models and found that Roberta-base, Roberta-large, Studio Ousia Luke-large, MS Deroberta-large worked better for the problem. Here are ourt best models and their oof rmse:

| model |  oof rmse|
| --- | --- |
| clrpdebertalargeppln4attheadwd | 0.47696|
| robertalarge-read2vec | 0.50651|
| litmodel-dataset | 0.47092|
| clrp-roberta-large-1h-att-head-readability-pt | 0.49076|
| lightweight-roberta | 0.47214|
| clrp-deberta-large-4-se-wd | 0.48815|
| litmodel-deroberta |  0.47007|
| clrp-roberta-large-2f-se | 0.48927|
| clrp-deberta-large-ppln4-atthead |  0.48395|
| clrprobertalargeppln4attheadwdr3l |  0.48865|
| clrp-deberta-large-4-se | 0.49059|
| litmodel-luke |  0.47827|
| deroberta-read2vec | 0.50054|
| clrp-roberta-large-2h-atthead-se | 0.49703|
| clrp-deberta-large-2-se |  0.48962|
| litmodel-roberta-large |  0.48182|
| roberta-large-outlier-trim | 0.50249|
| litmodel-deroberta2 |  0.47150|
| studioousialukelarge | 0.49649|

We tried pre-training, but the results didn't improve (probably we needed a larger and better selected dataset) so we gave up insisting on that.

3. We used just two different methods to deal with the output of our transformer models (CRLP and Lightweight as you can find in the public Kernels) using single or multiple attention heads, but we exercised our creativity in figuring out model heads. We found that piping some readability measures into the final dense layers helped the reslt a bit. We came up with the features to concatenate after doing a residual analysis on our off and finding out that they still correllated with indexes such as Kincaid, FleschReadingEase, GunningFogIndex, SMOGIndex and with simple measurements such as the number of syllables or the ratio of complex words on the total of words. We also tried to exclude certain examples from the training in order to obtain more diverse models.

4. We experimented with different learning rates and trying to freeze part of the layers: we got out the best results using the AdamW optimizer, having a warm-up epoch and then decreasing the learning rate and increasing it using the cosine scheduler. After the first epoch we also started mixing up the target, drawing for a part of it another value based on the standard error provided.

5.  Knowing that the best epochs were from 2 to 5, we often evaluated during this phase and retained the state of the model with best validation, even if the epoch wasn't completed.

6. After having a sufficient number of models we stacked them as features together with all the readability measurements provided by the readability package. We used a Ridge regression, a linear SVR and an ElasticNet, tuning both their hyperparameters and doing forward selection of the features to be included. The forward selection helped our final ensembles to be both different and well performing, getting better results that finding weights by hand. Finally we blended the three models to obtain our prediction. This move gave use the boost climb up in the last days from position 190th to 55th.

In conclusion, we indeed found this competition interesting in order to figure out what to do when you have a small textual datasets and noisy targets.
