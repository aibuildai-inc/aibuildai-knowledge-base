# [Solution Share] Convolutional Bi-LSTM with median and classic time series models (Python + R!)

Competition: web-traffic-time-series-forecasting
Rank: #16
Source: https://www.kaggle.com/c/web-traffic-time-series-forecasting/discussion/39887

My approach is a weighted ensemble of **Deep Learning** (`Keras` in Python), **median-based** (modified version of public kernels), and **ensembled classic time series forecasts** (`forecastHybrid` in R) models, which scores **~31.0** in the first 10 days of future time frame. I'm also amazed by the performance of deep learning model in this competition!

<br>

It was a `Convolutional Bi-LSTM network` (a stack of one convolutional 1D layer, one max-pooling 1D layer, and one bi-directional LSTM layer, with a linear output layer) for all pages. The model is rather simple, with only 16 filters of a kernel size of 3 in Conv1D, followed by a max pooling of size 2, and 64 neurons in each direction of bi-LSTM (128 neurons in total) with a `dropout` rate of 0.2. It used `MAE` as the loss function and `Nesterov Adam` as the optimizer.

About neuron activation types, it was `relu` for Conv1D and `tanh` for LSTM. Note that for kernel initializer of LSTM I selected `he_uniform` (**He uniform variance scaling initializer**) instead of the default `glorot_uniform` one. I didn't use any batch normalization because it produced worse result (although it should be mostly better to use it).

All the hyperparameters were tuned based on a random shuffled local validation set from 10% pages, also a local hold-out (last 62 days) for overfitting detection.

Adding more layers and neurons was not helpful and eventually caused overfitting.

The idea of using bi-directional LSTM was inspired by the solution shared by [Aaron Sim][1], the winner of [How Much Did It Rain? II][2] competition.

I trained the DL model three times with different random seeds, early stopping (stopped at `10-12 epochs`) and a `batch size` of 256, and then ensembled them to reduce prediction variance. Those DL models contributed a SMAPE score of ~37.7 in local hold-out.

Each of them took about ~2 hours to train on a 16-cores CPU, 24 GB machine, with `tensorflow-cpu` compiled with `AVX2` support for performance boost. One interesting thing is that 32-cores was not performing much better than 16-cores one! After this competition I would like to buy a `GTX-1070/1080` GPU to make my life easier...

Another important thing to this competition to make DL work well is the scaling of page views. After several experiments I applied`log1p` first then with `StandardScaler`, which gave the closet and consistent gap between local CV and public LB. Note that for StandardScaler I discovered that it is better to only normalize the mean but keep the std to have its own variance in the original time series. (`StandardScaler(with_mean=True, with_std=False)`).



<br>

As for the use of `forecastHybrid` in R, instead of training a model for all, I trained one model for each page, each of them is a combination of `ets` (**Exponential smoothing state space model**), `thetam` and `stlm` (**Seasonal and Trend decomposition using Loess model**). Since in some pages it may perform badly and caused the whole SMAPE to jump, I used last 62 days as the local hold-out with a threshold of 42.0 to decide whether to take the combined predictions of a page or not. If it's above the threshold then I replaced the prediction with the last 49 days of median views grouped by weekend. It was able to achieve ~41.2 in local hold-out of stage-2 training set.

Due to **auto-ARIMA** 's long training time and relatively less promising result, it was not selected into the combination. It took about ~15 hours to train all page models on a 32-cores CPU, 48 GB machine with the help of `mclapply` in R.

<br>

This is my first attempt to use DL model in Kaggle competition, learned a lot from the book "[*Hands-On Machine Learning with Scikit-Learn and TensorFlow: Concepts, Tools, and Techniques to Build Intelligent Systems*][3]" by Aurélien Géron to make all things happen. Highly recommended this book!

Will share more details in a formal write-up after the competition! (though not sure if I would survive in Top 10 after that :))


  [1]: https://www.kaggle.com/aaronsim
  [2]: http://blog.kaggle.com/2016/01/04/how-much-did-it-rain-ii-winners-interview-1st-place-pupa-aka-aaron-sim/
  [3]: https://www.amazon.com/Hands-Machine-Learning-Scikit-Learn-TensorFlow/dp/1491962291
