# 19th Place Solution

Competition: PLAsTiCC-2018
Rank: #19
Source: https://www.kaggle.com/c/PLAsTiCC-2018/discussion/75167

# Final stand
- Public LB: 0.838
- Private LB: 0.851

# Features
- aggregate
- focused on peak
- focused on detected
- Luminosity
- LC fitter

# Models
- LGB
- XGB
- MLP

# Post process
We used weighted multi logloss(same as kernel) and multi logloss.
After predicted by weighted multi logloss, calculated weight for each class using [gradient descent][1] .
For multi logloss, [here][2].

# Did not work
- AE
- Augmentation

![model pipeline][3]


  [1]: https://github.com/KazukiOnodera/PLAsTiCC-2018/blob/master/py/utils_post.py
  [2]: https://github.com/KazukiOnodera/PLAsTiCC-2018/blob/master/py/utils.py#L400-L406
  [3]: https://storage.googleapis.com/kaggle-forum-message-attachments/441823/10901/model_pipeline_compreessed.png
