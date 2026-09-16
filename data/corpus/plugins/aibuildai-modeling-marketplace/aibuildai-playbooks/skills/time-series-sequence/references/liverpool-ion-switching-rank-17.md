# 17th Private & 17th Public Place Solution

Competition: liverpool-ion-switching
Rank: #17
Source: https://www.kaggle.com/c/liverpool-ion-switching/discussion/153829

Congrats to all the kagglers who participated in this competition and especially to those researchers who were desperately trying to overcome 0.946 barrier on public LB 😊 

I am really enthusiastic and satisfied since it is our first kaggle competition.

We were using a dataset with [removed drift](https://www.kaggle.com/cdeotte/data-without-drift) and [Kalman filtering](https://www.kaggle.com/teejmahal20/a-signal-processing-approach-kalman-filtering).

Nearly 4% of the train data was cut as too noisy.

Our solution is based on [WaveNet](https://www.kaggle.com/siavrez/wavenet-keras) and [Wavenet with SHIFTED-RFC Proba and CBR](https://www.kaggle.com/nxrprime/wavenet-with-shifted-rfc-proba-and-cbr) backed by a callback that provided early stopping in case macro f1 would not grow in 40 epochs with further weights' rollback.

We have modified this estimator with an ensemble technique inspired by [Monte Carlo method](https://en.wikipedia.org/wiki/Monte_Carlo_method). The base prediction is rounding the medians of predictions.

An important step was to split data into 5 groups w.r.t. average open channels. This idea has led us to the fact that our WaveNet ensemble provides a lower bound for the group with the highest average number of open channels.

The final estimator looks like **max(predictions)** for the group with the high average number of open channels and **round(median(predictions))** for everything else.

We are going to publish a [full kernel](https://www.kaggle.com/biruk1230/17th-place-simple-wavenet-solution-ion-switching) shortly.
