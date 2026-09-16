# #19 Solution | AutoGluon + TF + LightGBM + XGBoost + Calibration

Competition: playground-series-s4e8
Rank: #19
Source: https://www.kaggle.com/c/playground-series-s4e8/discussion/531347

This was my first time using `AutoGluon`, and I must admit that I enjoyed its simplicity and great performance. In this post, I will explain my approach.

# Data Preprocessing
In terms of data processing, I considered three cases. 

- Preprocessing presented in [AmbrosM's notebook](https://www.kaggle.com/code/ambrosm/pss4e8-eda-which-makes-sense).
- Preprocessing presented in [Stephen Murphy notebook](https://www.kaggle.com/code/stephenmurph/clean-schrooms-for-autogluon-top-20)
- No preprocess 

# Modeling 
In terms of models, I built several models. The below table summarizes my results.
| Model | # of Models | Worst CV score| Best CV score | Ensemble CV score |
| --- | --- |
| `AutoGluon`| 20  | 0.98512 |  0.98525 | -  |
| `LightGBM`| 9 | 0.98423 |  0.98477 |  0.98484 |
| `XGBoost`| 7 | 0.98417 |  0.98485 | 0.98488  |
| `TensorFlow`| 6 | 0.98383 |  0.98401 |  0.98452 |

Note that the above results are based on a 10-fold cross validation strategy.

# Ensemble 
In terms of ensemble, I did the following: 

```
0.62 x AutoGluon (best model) + 0.38 x AutoGluon of (LightGBM Ensemble, XGBoost Ensemble, TensorFlow Ensemble)
```
The above ensemble obtained 0.98527 CV score over 10-folds.

# Calibration
Finally, I calibrated the predictions with `IsotonicRegression`, which boosted the 10-fold CV score by 0.00003

# What did not work
I run a few experiments in which I optimize the threshold to come up the label, but I couldn't find consistent results.
