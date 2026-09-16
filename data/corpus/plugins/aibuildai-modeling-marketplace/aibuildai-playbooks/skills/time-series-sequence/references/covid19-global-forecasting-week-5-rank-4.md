# 4th Place Neural Network Solution

Competition: covid19-global-forecasting-week-5
Rank: #4
Source: https://www.kaggle.com/c/covid19-global-forecasting-week-5/discussion/154664

I think the models we build are too late for being useful in this pandemic. They can hopefully (but not hopefully) be used in another pandemic. Therefore I wanted to have a re-usable Machine Learning solution. So far, it performs okay but it is early to talk about it. Anyway, I wanted to explain it before I forget the details. I have 2 notebooks, one for training and one for inference:
https://www.kaggle.com/aerdem4/covid-19-w5-training
https://www.kaggle.com/aerdem4/covid-19-w5-pipeline

It is CNN Model trained on last 2 weeks of data. In last covid-19 competitions, my NN had diverged by time. I have learned from my mistakes and didn't want to do recursive predictions for this one. I have trained different model for each n day ahead. Here is the model architecture:



Some tricks that worked:
- Cumulative sum over relu as the last layer guarantees that I have non-zero predictions and quantiles are always in order.
- Training directly with pinball loss. In initial epochs, I have trained for 0.15, 0.50 and 0.85 quantiles. Then late epochs are trained with actual quantiles (0.05, 0.50, 0.95). Without this, model starts being cautious very early and gets stuck in a local minimum.
- Bagging 5 NN models for each day to reduce randomness
- Smoothing predictions for each day n: `y(n) = 0.2*y(n-1) + 0.6*y(n) + 0.2*y(n+1)`
- Rounding predictions to integers. This probably helps a little with zero cases.
