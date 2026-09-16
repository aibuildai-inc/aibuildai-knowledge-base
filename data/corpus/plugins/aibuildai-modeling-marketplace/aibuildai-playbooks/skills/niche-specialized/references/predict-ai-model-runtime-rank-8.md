# 8th place solution

Competition: predict-ai-model-runtime
Rank: #8
Source: https://www.kaggle.com/c/predict-ai-model-runtime/discussion/456645

Thanks to competition organizer giving us such a interseting competition.Our team joined very late we don't have time to deepdive but we found a simple solution very useful.

# Data Processing & Feature Engineering
Tile: count of config&node&edge, mean,max,std,last of node_feat, mean,max,std of config_feat
Layout: flatten node_config_feat -> remove unique value columns -> remove duplicate columns

# Train Data
The keypoint of our layout solution is finding the most similar train data for each test data. We can observe some data have almost same edge&node number and can guess they are same model type except size,the test data should be the same model with different batch size.
e.g.
```python
train: small_bert_bert_en_uncased_L-12_H-768_A-12_batch_size_16_test
valid: small_bert_bert_en_uncased_L-12_H-768_A-12_batch_size_32_test
test(same edge&node number) should be the small_bert_bert_en_uncased_L-12_H-768_A-12_batch_size_64_test
```
We cannot find all the similar train data for each test data, but enough to get a good result.And it's very fast for iteration.

# Model
transform the target to minmaxscaler , use xentropy as the loss function, lightgbm as model, we don't use validation just give a fix round for each model.
