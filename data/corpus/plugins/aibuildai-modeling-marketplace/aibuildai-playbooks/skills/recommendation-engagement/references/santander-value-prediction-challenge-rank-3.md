# 3rd place

Competition: santander-value-prediction-challenge
Rank: #3
Source: https://www.kaggle.com/c/santander-value-prediction-challenge/discussion/63864

**Column and row groups.** After the leak was published (thanks to [Giba][1] and [Mohsin Hasan][2]) as many of participants I started to search other groups of columns and groups of rows. Of course, I wanted to find them all :) I selected a column or a row and then searched next or previous (mainly by script but also I looked through result and correct some mistakes manually). For rows search I assumed that rows with target == 0 were eliminated from train and test. So there can be gaps. For columns search I assumed that group of columns consists exactly of 40 columns. Maximum length of group of rows was 116, but there were shorter row groups and groups with big gaps. Of course, I tried to find as much groups of columns as possible on given groups of rows and if I couldn't find new group of columns I started groups of rows search and then returned to columns' search and so on. At the end I also used groups of rows from test.

My best submission includes gbm models based on set of 97 and 110 groups of columns. I tried to find all groups (125) but either I made mistakes or due to randomness of data blendings which contain gbm models based on 125 groups of columns were worse than those on smaller number of groups. Last ten groups (116-125) were shorter than 40 and with gaps.

I selected the best submission by public score and the forth. On private the forth became the best and the best became the second.

**Augmentation.** Leak values found in test I used for augmentation train for training gbm models. Based on 97 groups of columns I found 3 894 leaks in train (99.85% of them were correct) and 7 844 leaks in test at lag # 35. Based on 110 groups of columns I found 3 898 leaks in train (99.8% of them were correct) and 7 954 leaks in test at lag # 37. I wasn't sure on leaks at lags 36 and 37 (on train there wasn't leaks at lag 36 and 2 of 2 error leaks at lag 37).

**Features** for gbm:

 - Raw columns, columns from main group (which starts from column f190486d6)

 - Log of mean, mean of log, log of median, log of max, log of min, number of zeros, sum, std, kurtosis, skew, number of uniques calculated on all columns and on each found group separately

 - New group of columns, which was sum of three most important groups by feature importance (which start from columns f190486d6, df838756c and b6daeae32) and features from previous point calculated on this new group

 - New groups of columns, which were sum of some pairs of groups among the most important groups by feature importance (It didn't give good cv increase so without details)

 - I applied feature selection based on gbm feature importance and parameters tuning based on 4 fold validation

 - Final submission was weighted sum of 23 models. Most of models were made with lightgbm and some with xgboost

 - Weights were adjusted iteratively on oof train+leak prediction. As I didn’t use cv here and to avoid weights overfitting firstly I accepted only those weights changes which lead to rmsle decrease on more than X, but in my best submission I accepted every weight change (because among similar submissions it gave better public score).

  [1]: https://www.kaggle.com/titericz/the-property-by-giba
  [2]: https://www.kaggle.com/tezdhar/breaking-lb-fresh-start
