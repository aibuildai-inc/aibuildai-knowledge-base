# 37th place solution (Transformer part)

Competition: riiid-test-answer-prediction
Rank: #37
Source: https://www.kaggle.com/c/riiid-test-answer-prediction/discussion/209624

First of all, I would like to thank oganizer and Kaggle for hosting the competition. In addition, thanks for @vk00st inviting us to merge as a team, and other teammates @unkoff @arvissu @syxuming, we have great team work.

I also want to thank @manikanthr5 @wangsg @leadbest great SAKT starter notebook. I can’t implement the Transformer model by myself without these.

I summarize the detail of our SAINT+ like model as follows.


## Input features

1. content_id
2. response (answered_correctly)
3. part
4. prior_question_elapsed_time
5. prior_question_had_explanation
6. lag_time1 - convert lag time to seconds. if lag_time1 >= 300 than 300.
7. lag_time2 - convert lag time to minutes. if lag_time2 >= 1440 than 300 (one day).
8. lag_time3 - convert lag time to days. if lag_time3 >= 365 than 365 (one year).

Lag time split to different time format boosting score around 0.003.

## Distinguish zero padding
* In order to distinguish zero padding and zero of **content_id**, I add one to **content_id** and **prior_question_had_explanation** and **response**. It will help model distinguish zero padding and features. It boost score around 0.003.

```
q_ = q_+1
pri_exp_ = pri_exp_+1
res_ = qa_+1
```
### Transformer

##### Encoder Input
* question embedding
* part embedding
* position embedding
* prior question had explanation embedding

##### Decoder Input

* position embedding
* reponse embedding
* prior elapsed time embedding
* lag_time1 categorical embedding
* lag_time2 categorical embedding
* lag_time3 categorical embedding
* Note that I tried categorical and continuous embedding in prior elapsed time and lag time. The performance of categorical embedding is better than continuous embedding.

##### Parameter of  Transformer
* max sequence: 100
* d model: 256 
* number of layer of encoder: 2
* number of layer of decoder: 2
* batch size: 256
* dropout: 0.1
* learning rate: 5e-4 with AdamW

### Blending

* LGBM + Catboost + three Transformer
* LGBM + two Transformer

### Result

* Finally, our team get LGBM public LB 0.794 and Transformer public LB 0.799.
* Get public LB 0.804 with blending.

### Sample code
* [Training](https://www.kaggle.com/m10515009/saint-is-all-you-need-training-private-0-801)
* [Inference](https://www.kaggle.com/m10515009/saint-is-all-you-need-inference-private-0-801)
