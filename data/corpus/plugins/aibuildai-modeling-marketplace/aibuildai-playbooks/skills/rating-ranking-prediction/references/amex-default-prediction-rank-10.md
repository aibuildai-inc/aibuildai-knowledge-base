# 10th Place Solution: XGB with Autoregressive RNN features

Competition: amex-default-prediction
Rank: #10
Source: https://www.kaggle.com/c/amex-default-prediction/discussion/347668

What a competition! I really enjoyed it and only hope I could have found more time. First of all, I would like to thank Raddar @raddar, Martin @ragnar123, and many others who generously shared codes and datasets! The public solutions are of amazing quality. And it also determines my game plan: **create something original and blend it with the best public solution.** My solution is based on RAPIDS cudf for dataframe processing, XGB for training, and pytorch lightning for feature extraction.

There are two vital observations of this dataset:
- the big test data which is from the future is available.
- short sequences (<13) are culprits of bad performance.

Let's start with the latter observation:**the sequence length of each customer profiles** plays a critical role in the model performance:
```
import cudf
path = '/raid/amex'
train = cudf.read_parquet(f'{path}/train.parquet',columns=['customer_ID'])
trainl = cudf.read_csv(f'{path}/train_labels.csv')
train = train.merge(trainl,on='customer_ID',how='left')
train['seq_len'] = train.groupby('customer_ID')['target'].transform('count')
train = train.drop_duplicates('customer_ID',keep='last')
train.groupby('seq_len').agg({'target':['mean','count']}).sort_index(ascending=False)
```
output:
```
           target
       mean	count
seq_len		
13	0.231788	386034
12	0.389344	10623
11	0.446737	5961
10	0.462282	6721
9	 0.450164	 6411
8	 0.447300	 6110
7	 0.418430	 5198
6	 0.387670	 5515
5	 0.392635	 4671
4	 0.416221	 4673
3	 0.358602	 5778
2	 0.318465	 6098
1	 0.335742	 5120
```
It is obvious that **sequence length 13** is the most common but also with a significantly lower mean default rate. At first glance, I thought it meant shorter sequences are easier to predict since they have more positive samples. But I'm quickly proven wrong when checking my cross-validation results:
```
Fold 0 amex 0.7990 logloss 0.2144
Fold 0 L13 amex 0.8214 logloss 0.1928
Fold 0 Other amex 0.6724  logloss 0.3289
```
The 1st line is the overall score. The 2nd line is the score of sequences of length 13 and the 3rd line is the score of all the rest sequences. Apparently, shorter sequences have a much worse score than the full sequences of length 13. This is also an implication of how the short sequences are truncated: the more recent profiles are deleted, which could explain the big degradation of the score because more recent profiles have more predicting power in general.  For example, let's say for 13 consecutive months (M1~M13) and sequence A is of length 13 and sequence B is of length 8:
```
   M1 M2 M3 M4 M5 M6 M7 M8 M9 M10 M11 M12 M13
A  1  1  1  1  1  1  1  1  1  1  1  1  1  1 
B  1  1  1  1  1  1  1  1  1  0  0  0  0  0 
``` 
where `1` means features exist and `0` means features missing. Of course, there is another possibility:
 ```
   M1 M2 M3 M4 M5 M6 M7 M8 M9 M10 M11 M12 M13
A  1  1  1  1  1  1  1  1  1  1  1  1  1  1 
B  0  0  0  0  0  1  1  1  1  1  1  1  1  1 
``` 
We can actually find out which one is more plausible by `unstacking` dataframes in the above two ways and run xgboost with them, respectively. As expected, the former has a better CV score which indicates it is likely how the truncation of short sequences is done.

If we could somehow predict the missing profiles of sequence `B`, the life of the downstream XGB models would be made much easier. An intuitive choice is to generate the missing profiles using a one-dimension auto-regressive RNN. Bascially we want to predict the features of the next month based on the feature values of the current month and all previous months. And when we have the prediction for the next month, we can use it as part of the input and predict again and so on so forth. This is also where the availability of the big test data really shines. Since we are predicting features, not `target`, we can train our models using both `train` and `test` data. The RNN structure is very simple:  just one GRU layer and some FC layers. The RNN performance is pretty decent. In terms of RMSE of all 178 numerical features, the GRU achieves validation `RMSE 0.019`. For simplicity, all features are log-transformed and `fillna(0)`. You might wonder how good is `RMSE 0.019`. We can simply compare it with the naive baseline: just repeat the last available month. For example, if I'm asked to predict features of M2, the naive baseline is just output features of M1. The RMSE of this naive baseline is `0.03` so our RNN actually learns something and could be useful.

The rest would be straightforward, after predicting missing months, now every sequence is of length 13 so I just unstack the dataframe to increase the number of features 13x. For example, instead of having one feature `P_2` of the last month, now we have 13 features `P_2_M_1` to `P_2_M_13`. These are the most useful features I created. For downstream classifiers I only use XGB so that it is *not similar* to [the great LGB DART notebook](https://www.kaggle.com/code/ragnar123/amex-lgbm-dart-cv-0-7977). By varying RNN hyperparameters, XGB hyperparameters and different combination of features, I end up with 7 XGB models, whose ensemble is  0.7993 CV and 0.799 public LB. Averaging it with the best public solution and with extraordinary luck, my final submission ended up in the gold zone.



The final thought is my best auto-regressive features are generated 1 hour before the deadline. I'm very happy it worked!
