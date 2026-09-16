# 3rd place - 135 OOFs

Competition: playground-series-s6e1
Rank: #3
Source: https://www.kaggle.com/c/playground-series-s6e1/writeups/3rd-place-135-oofs

Congrats to all the winners and special thanks to all the people who shared insight and notebooks for this competition: @cdeotte, @include4eto, @siukeitin, @mikhailnaumov, @omidbaghchehsaraei, @mahoganybuttstrings, @yekenot.

My solution is pretty much the same as @tilii7's solution: just add as many diverse models to an ensemble. I didn't really bother much with FE (I was kinda too lazy to test anything :P).

#Models

Here are my models:

| Model | CV | Private LB |
| --- | --- | --- |
| XGB | 8.59994 | 8.59171 |
| LGBM | 8.62145 | 8.62143 |
| CatBoost | 8.69769 | 8.66484 |
| TabM | 8.58894 | 8.58800 |
| RealMLP | 8.59069 | 8.58555 |
| LNN | 8.61888 | 8.61536 |

My CatBoost is bad cuz I didn't really work on it as much and it ran way too slow (maybe I set the hyperparameters wrong or something).

Some other public notebooks:

https://www.kaggle.com/code/mahoganybuttstrings/pg-s6e1-xlearn-ffm-cv-8-66086-lb-8-62539

https://www.kaggle.com/code/mahoganybuttstrings/pg-s6e1-lnn-cv-8-62289-lb-8-57507

https://www.kaggle.com/code/omidbaghchehsaraei/xgb-predicting-student-scores-cv-8-64161

https://www.kaggle.com/code/omidbaghchehsaraei/autogluon-predicting-student-scores-lb-8-5767

https://www.kaggle.com/code/omidbaghchehsaraei/tabm-predicting-student-test-scores-cv-8-61131

https://www.kaggle.com/code/omidbaghchehsaraei/resnet-predicting-student-test-score-cv-8-63409

https://www.kaggle.com/code/omidbaghchehsaraei/ft-transformer-predicting-student-score-lb-8-56872

https://www.kaggle.com/code/yekenot/ps-s6-e1-deeptables-nn

My final ensemble (Ridge) has CV 8.57299 and Private 8.57775
