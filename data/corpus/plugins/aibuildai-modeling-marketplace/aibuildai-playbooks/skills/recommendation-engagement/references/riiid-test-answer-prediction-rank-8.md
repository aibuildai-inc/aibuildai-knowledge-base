# #8 solution: Ensemble of 15 same NN models

Competition: riiid-test-answer-prediction
Rank: #8
Source: https://www.kaggle.com/c/riiid-test-answer-prediction/discussion/210552

Hi all,
Learned a lot from other's solutions! In this post, I would like to share some insights in my solution.

In summary, my best single NN model could achieve 813/815 public/private score, by ensemble of 5 folds and 3 snapshots in each fold, finally 15 nn models achieve the 814/816 public/private score.

With 15 nn models, the online inference cost **less than 4 hours**, thus it can ensemble at least 30 models in this pipeline.

## Dataset split

I split the train and valid set by several steps:

1. Calculate all unique users in dataset.
2. Select 5% users totally in the valid set, and 45% users totally in the train set.
3. For the least 50% users, random split the user data into train and valid set by time.

Thus we have both individual user in train and valid set, and also have many users who appear in both train and valid set. This split can achieve less than 0.001 score difference compared with LB.

By changing different random seed, we can get different folds.

## Feature engineering

Since there are many detailed FE in other's post, I will just share some important points.

### Basic features

1. Evaluate user ability

   + We can evaluate user ability by his history action, including the correctness, time elapsed, lag time, and so on.
   + These features can be calculated on not only content level, but also on same part, same tags, same content and so on.
   + These features can also be extended based on time, for example, we can make a feature which is the user correctness in last 60 seconds.

2. Evaluate content difficulty

   + We can evaluate content difficulty by calculate its global accuracy, std, average time elapsed and so on.

3. Evaluate User x Content features

   Even a content is difficult, the user may still skill enough and can solve it correctly. Thus we also need to describe how the user could perform on this content.

   + user_acc_diff: For a content, if the content has low global acc, but the user answered it correctly, the user might above the average of all users. We can use the $logloss(content_global_average_acc, user_answer)$ to evaluate the difference.

   + user_elapsed_diff: like the user_acc_diff, we can also evaluate user time elapsed difference in user history contents.

We can dig many features on above three fields, for example, user's average lag time on history content/ history same part content/ history same tag content.. could also be a useful features.

### Other features/tricks

1. Time related features: last_content_timestamp_diff, last_lag_time and its statistical information in history.
2. Abnormal Users: If a user answered every content less in 4 seconds, and all his chose answer are same (such as C), then if the correct answer of next content is C, we can believe he will correctly answered next content.
3. Learned lectures for a specific content: If there is a content-lecture-content pattern in user history, and the two content are same contents. It might the user learned a specific lecture for this content, which means in the second time, the probability he answered it correctly is high.
4. Wrong answer ratio: There might be a pattern like "select C as default for hard contents".  Thus calculating the ratio of user choice on his incorrectly content can tell as weather the user could lucky guess current content even he don't know he correct answer.

Also, features such as current content id/current timestamp/current part are also used. Finally, I get 120 dim features, which can get public 0.806 by a single lgb model in single fold.

P.S. The categorical feature (set on content id) of lightgbm can boost my score about 0.003.

There are also many useful hint which could improve the speed and save the memory:

1. Do not use pandas to calculate features, transfer it to numpy or just python.
2. Using a class to store user information, and save each user information in individual file.
3. In the inference phase, we can only read the user information for who appeared in the test set, the total number of users in test set are less than 10,000. This is the key to reduce memory. If the memory still not enough, the LRU-cache could be used to remove unused users.
4. In my experiment, using numpy array to store information cost more disk space compared with python variable.

## NN

Since it is very late when I notice the key to the top is NN model, I don't have much time analysis the NN models, especially design a specific features or structures. To save the time, I use the lightgbm features as the NN input (Time axis is added and seq len is 128).

#### Robust Standard Normalization

 There are many outlier in the features from lightgbm, thus simply utilize standard normalization can hardly get desired results. I utilized the robust standard normalization to normalize all the features:

```
def robust_normalization(column):
	cur_mean = np.nanmedian(column)
    cur_qmin, cur_qmax = np.nanpercentile(cur,[2.5, 97.5])
    cur_std = np.nanstd(column[(column>=cur_qmin) & (column<=cur_qmax)])
    column = np.clip(column, a_min = cur_qmin, a_max = cur_qmax)
    column = (column-cur_mean)/cur_std
    return column
```

#### Models



My NN model is very simple, just a transformer encoder with a fc classifier. The encoder has 4 transformer layers and embed dim is 128, no other modifications. The size of model file is only 7M, that's why I can ensemble many models on inference phase.

The model can achieve 813/815 on single model/single fold. and 814/816 on 5 folds ensemble. There are much time left on inference, I use 3 snapshot ensemble on each fold, which can also boost some scores.
