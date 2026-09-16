# 14th place solution / Single model with different random_states Voting + blending

Competition: tabular-playground-series-sep-2021
Rank: #14
Source: https://www.kaggle.com/c/tabular-playground-series-sep-2021/discussion/276141

The main point was to  use the same models with the same parameters but `random_state` as estimators for VotingClassifier. The idea is described in this [notebook](https://www.kaggle.com/martynovandrey/one-model-voting-from-0-81800-to-0-81837).  It was the key to increase the scores of single models.

First there was two notebooks with traditional  method [LGBM model](https://www.kaggle.com/martynovandrey/tps-september-lgbm) with score **0.81802** and CatBoost model with **0.81751**

Using `single model voting` the scores improved to **0.81839** [LGBM single model voting](https://www.kaggle.com/martynovandrey/one-model-voting-from-0-81800-to-0-81837) and to **0.81816** [CatBoost single model voting](https://www.kaggle.com/martynovandrey/one-model-catboost-voting)

The result were averaging with weights [0.7, 0.3], the score **0.81846**

It was blended with submission of [[TPS-09] Simple Blend & Stacking (XGB, LGBM, CATB)](https://www.kaggle.com/mlanhenke/tps-09-simple-blend-stacking-xgb-lgbm-catb) by [mlanhenke](https://www.kaggle.com/mlanhenke) (Thanks for sharing!), the score became **0.81854**

Then I modified the [[004-2o] lightGBM colsample TPS-sep-2021](https://www.kaggle.com/ivankontic/004-2o-lightgbm-colsample-tps-sep-2021) notebook by [Ivan Kontic](https://www.kaggle.com/ivankontic) (Thanks!) with **0.81835** score and improve the score using `single model voting` to **0.81845**

Blendig with weights [0.7, 0.3] gave the final score **0.81868** (**0.81752** private). The place was decreased by 3, IMHO because of +62, +37 and +15.

For me the main result was that the idea of single model voting do works in classification. 

Congratulations to the winners and thanks to [Edrick Kesuma](https://www.kaggle.com/edrickkesuma) and many others for interesting discussions and usefull ideas.

Many thanks to all those who upvoted [One model Voting](https://www.kaggle.com/martynovandrey/one-model-voting-from-0-81800-to-0-81837), it's my first silver 😊

* *Some notebooks executed on local PC, so the scores above may differ with notebooks shared.*
