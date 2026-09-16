# #5 Solution

Competition: tabular-playground-series-apr-2022
Rank: #5
Source: https://www.kaggle.com/c/tabular-playground-series-apr-2022/discussion/322277

Firstly, congratulations to the competition winners!

My solution focused on a singular GBDT (LGBM) model using features generated from tsfresh. Then for each feature generated with tsfresh I create matching features using the normalized feature value by subject. The ~9000 generated features were then trimmed down using recursive feature elimination. My final LGBM model had a 0.97816 private LB score and a 0.98248 public LB score.

I then simply blended my LGBM model (weighting ~0.4), with my LSTM model (0.98259 private LB,
0.98131 public LB) (weighting ~0.2). The remaining weights (~0.4) were split between:

- https://www.kaggle.com/code/dlaststark/tps-apr22-tfv1 
- https://www.kaggle.com/code/dlaststark/tps-apr22-tfv2 
- https://www.kaggle.com/code/dmitryuarov/sensors-deep-analysis-0-98 
- https://www.kaggle.com/code/bannourchaker/deep-learing-part2-bilstm-densenet-rnn-con6
- https://www.kaggle.com/code/bannourchaker/deep-learing-part3-cnn-inceptiontime-con11
- https://www.kaggle.com/code/bannourchaker/deep-learing-part4-hybrid-cnn-lstm-parallel-con166 
- https://www.kaggle.com/code/hasanbasriakcay/tpsapr22-fe-pseudo-labels-bi-lstm 

Big thanks to the authors of these notebooks for sharing their work!
