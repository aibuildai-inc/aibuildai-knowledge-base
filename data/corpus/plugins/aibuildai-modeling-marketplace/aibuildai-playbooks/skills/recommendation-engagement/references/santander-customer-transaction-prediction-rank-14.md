# 14th solution and magic feature(double peak)

Competition: santander-customer-transaction-prediction
Rank: #14
Source: https://www.kaggle.com/c/santander-customer-transaction-prediction/discussion/88927#latest-515803

Congrats to the winners. This competition is very interesting.
Here is my code.
https://github.com/qrfaction/Kaggle_SCTP_gold_14th_solution
My summary will be updated soon

update:
Because my English is not well, I don't know how to explain my features in English. 
By grouping the value count, I found double peak in "count==1" group of many features.
Here is the main code.

```
train[col + 'c1'] = (train[col + 'value_count'] == 1)
train[col + 'isP1'] = (train[col] &lt; train[col + 'm'])
train[col + 'p1dis'] = (train[col] - train[col+'p1']) * train[col + 'isP1'] * train[col + 'c1'] + (train[col] - train[col+'mean']) * (1-train[col + 'c1'])
train[col + 'p2dis'] = (train[col] - train[col+'p2']) * (1-train[col + 'isP1']) * train[col + 'c1'] + (train[col]-train[col+'mean']) * (1-train[col + 'c1'])
```

train[col + 'isP1'] is peak1 mask.
train[col+'p1'] is the peak1 point
train[col+'mean'] is the mean of each count.

I created three groups of features for ensemble in this way.
These are the difference between me and other teams.
My model has gained a lot from this feature.
LGBM:  0.92475 public /0.92367 private using data augmentation
