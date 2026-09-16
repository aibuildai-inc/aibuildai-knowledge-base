# #2 solution

Competition: predicting-red-hat-business-value
Rank: #2
Source: https://www.kaggle.com/c/predicting-red-hat-business-value/discussion/23824

Solution consists of three parts.

1. For the groups presented in the training sample I constructed some probabilistic interpolation model.
It may be explained by the following illustration.
The next plot shows the predictions (target probabilities) for the group 7 over the dates.
![enter image description here][1]

For the dates occurred in the training sample the probability is 1, for another dates the probability decreases.

The next plot shows the same predictions, but for the whole dates range.
![enter image description here][2]

2. We have 34224 different groups in the data. This number is the actual size of the sample, because just the groups are the objects from statistical point of view.

The only problem is that the features take different values inside the groups. For each group and for each feature I calculated a histogram. The bins of the histograms are the new features. This may be called as “fuzzy” version of binary encoding.

The prediction is a mixture of three models:

A) logistic regression;

B) kNN;

C) XGBoost based public scripts.

3. The last part is the leaderboard feedback.


  [1]: https://www.kaggle.com/blobs/download/forum-message-attachment-files/4893/f1.gif
  [2]: https://www.kaggle.com/blobs/download/forum-message-attachment-files/4894/f2.gif
