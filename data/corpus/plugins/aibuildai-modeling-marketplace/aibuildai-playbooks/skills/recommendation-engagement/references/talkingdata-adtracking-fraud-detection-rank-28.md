# 28 New Features - 0.9803 LB Score [Updated - v5]

Competition: talkingdata-adtracking-fraud-detection
Rank: #28
Source: https://www.kaggle.com/c/talkingdata-adtracking-fraud-detection/discussion/55030

**Update - May 04th** 

- Added 6 more features and removed few under performing features (28 new features in total) and the local validation score 0.99157 and the LB score improved to 0.9803.
Ensemble with another model of mine gives LB score of 0.9806.

Giving one last try..... adding 2 new features and removed 3 features

**Update - April 26th** 

- Added 5 more features (25 new features in total) and the local validation score jumped to 0.99145 and the LB score improved to 0.9802.

**Update - April 25th**

- Added 7 more features (20 new features in total) and the local validation score jumped to 0.991275 but sadly the LB score improved just to 0.9801.
- Looking to work on the feature importance to remove some under performing features and add some new features.

**Update - April 23rd**

Added 4 more features and the local validation jumped to 0.990979. The LB score also improved to 0.9800.

Apart from the base features, added 9 new features to get a LB score of 0.9798.

train size:  182403889
valid size:  2500000 (tail rows)

Training until validation scores don't improve for 50 rounds.
[10]    train's auc: 0.971521   valid's auc: 0.978039
[20]    train's auc: 0.977205   valid's auc: 0.981515
[30]    train's auc: 0.979879   valid's auc: 0.984628
[40]    train's auc: 0.981305   valid's auc: 0.987058
[50]    train's auc: 0.982025   valid's auc: 0.987659
[60]    train's auc: 0.982491   valid's auc: 0.988301
[70]    train's auc: 0.982942   valid's auc: 0.988885
[80]    train's auc: 0.983197   valid's auc: 0.989289
[90]    train's auc: 0.983447   valid's auc: 0.989419
[100]   train's auc: 0.983683   valid's auc: 0.989379
[110]   train's auc: 0.98386    valid's auc: 0.989594
[120]   train's auc: 0.983984   valid's auc: 0.989651
[130]   train's auc: 0.984101   valid's auc: 0.989784
[140]   train's auc: 0.984219   valid's auc: 0.989845
[150]   train's auc: 0.984317   valid's auc: 0.989851
[160]   train's auc: 0.984398   valid's auc: 0.990049
[170]   train's auc: 0.984523   valid's auc: 0.99027
[180]   train's auc: 0.984617   valid's auc: 0.990348
[190]   train's auc: 0.984713   valid's auc: 0.990424
[200]   train's auc: 0.984799   valid's auc: 0.990447
[210]   train's auc: 0.984853   valid's auc: 0.990475
[220]   train's auc: 0.984915   valid's auc: 0.990588
[230]   train's auc: 0.984979   valid's auc: 0.990649
[240]   train's auc: 0.985035   valid's auc: 0.990668
[250]   train's auc: 0.985089   valid's auc: 0.990664
[260]   train's auc: 0.985126   valid's auc: 0.990703
[270]   train's auc: 0.985166   valid's auc: 0.990704
[280]   train's auc: 0.985209   valid's auc: 0.990698
[290]   train's auc: 0.985251   valid's auc: 0.990775
[300]   train's auc: 0.985285   valid's auc: 0.990797
[310]   train's auc: 0.985315   valid's auc: 0.99079
[320]   train's auc: 0.985351   valid's auc: 0.990758
[330]   train's auc: 0.985379   valid's auc: 0.990737
[340]   train's auc: 0.985418   valid's auc: 0.990764
[350]   train's auc: 0.98545    valid's auc: 0.99076
Early stopping, best iteration is:
[307]   train's auc: 0.985306   valid's auc: 0.990801

Using 32GB Machine with 100GB Swap Space on 2 SSD's. I usually starts the script at night and wake up in the morning to see the beautiful results :)

PS: 7 Features gave me a LB score of 0.9788.

The next step is to add 3 more new features and see if it can get a score of 0.98XX
