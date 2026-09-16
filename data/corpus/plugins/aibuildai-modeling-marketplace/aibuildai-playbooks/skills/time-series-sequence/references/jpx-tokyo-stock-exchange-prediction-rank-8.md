# 10th Place - Simple Features + LGBM

Competition: jpx-tokyo-stock-exchange-prediction
Rank: #8
Source: https://www.kaggle.com/c/jpx-tokyo-stock-exchange-prediction/discussion/361127

This result has been a big surprise to me. Mainly because I had to stop working on it half-way through due to holiday and lack of time. 

Anyway, the notebook can be found [here](https://www.kaggle.com/code/vuk1998/jpx-submission-template/notebook?scriptVersionId=95590449).

You can see that the model is very basic. There are some simple features such as various returns (open->close, close->close), amplitude, volatility, moving averages etc.

Locally, I performed experiments (walk forward CV) to come up with the lgbm model attached.

Additionally, since I was using some aggregate features I also came up with some sensible default values until they become available.


Thanks to organisers for organising this!
