# #4 solution

Competition: tabular-playground-series-apr-2022
Rank: #4
Source: https://www.kaggle.com/c/tabular-playground-series-apr-2022/discussion/322558

The best single model appears to be based on LSTM layers; i tried to create my own, but it was worse than best public LSTM models, so i ended up using public LSTM models instead of my own ones (thanks to everybody who developed and published them).

What LSTM models seem to be missing is information aggregated by subject (all LSTM models look at 1 sequence at a time, so are missing all the subject info).

To get around this limitation i constructed lightGBM model with inputs aggregated by subject as well as by sequence, and fed predictions from best LSTM models as one of the features (also aggregated by subject in many ways). For that i had to rerun LSTM models to capture their prediction for train data (out of fold) in addition to prediction for test data.

This is it. This form of model stacking seems to produce better synergy than simple blending.
