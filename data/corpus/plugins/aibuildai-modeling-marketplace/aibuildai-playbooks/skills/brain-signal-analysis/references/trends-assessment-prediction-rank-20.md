# 20th place solution

Competition: trends-assessment-prediction
Rank: #20
Source: https://www.kaggle.com/c/trends-assessment-prediction/discussion/163557

Hi Everyone,

Ending final validation, I share my solution.

First, Channel-wise 3D convolutions are applied to fMRI spatial maps. All channels share the weight of convolution to prevent overfitting. Output features are thrown into Edge Update GNN with FNC correlation.

The outputs form GNN are averaged and concatenated with sMRI loading. Finally, conventional MLP is applied and prediction for age and other target variables are obtained.

Code: https://github.com/toshi-k/kaggle-trends-assessment-prediction

