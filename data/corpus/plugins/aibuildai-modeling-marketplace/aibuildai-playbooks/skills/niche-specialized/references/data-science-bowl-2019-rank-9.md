# 9th place solution

Competition: data-science-bowl-2019
Rank: #9
Source: https://www.kaggle.com/c/data-science-bowl-2019/discussion/127612

First of all, I would like to thank Booz Allen Hamilton and Kaggle for hosting such an interesting competition! I moved up from about 800th place to 50th place in 9 days before the end of the competition, and I moved up from about 80th place to 23rd place on the final day of the competition. So this week was very thrilling for me.

My solution was very simple. The summary is below.
1. Feature engineering (almost aggregation features)
2. Make some diverse models and stacking
3. Threshold tuning with random search

I introduce my solution's key points briefly.
# How to make train and test dataset
Many kernels used `get_data` function that deals with the user behaviour data sequentially.But I thought this function made it difficult for me to make and manage features.So I made a new approach for making train dataset. Let me show this. At first, to each game session, I assigned the number of assessment that a user tried until that game session . Below is an example.
```
train_gs_assess_dict = {}
for ins_id, user_sample in tqdm(train.groupby('installation_id')):
    assess_count = 0
    for gs, session in user_sample.groupby('game_session', sort=False):
        if session['type'].iloc[0] == 'Assessment':
            assess_count += 1
        train_gs_assess_dict[gs] = assess_count
train['assess_count'] = train['game_session'].map(train_gs_assess_dict)
```
Then I calculated the aggregation features for the subset of user activities before the assessment. The duration for creating train and test dataset became longer than kernel's. But this made implementation and management of features very easy.

# Model and stacking
I created the 8 models at first level.
|model|type|target|eval metrics|corr with accuracy group|kendall's tau|
|:---:|:---:|:---:|:---:|:---:|:---:|
|LightGBM|gbdt|accuracy group|rmse|0.621|0.460|
|LightGBM|goss|accuracy group|rmse|0.568|0.433|
|LightGBM|dart|accuracy group|rmse|0.619|0.459|
|LightGBM|gbdt|accuracy|rmse|0.615|0.457|
|LightGBM|gbdt|accuracy group&gt;2| auc|0.598|0.452|
|LightGBM|gbdt|accuracy group&gt;1| auc|0.615|0.456|
|LightGBM|gbdt|accuracy group&gt;0| auc|0.597|0.441|
|NN|-|accuracy group|rmse|0.600|0.444|

And I used Ridge Regression for stacking.
|model|type|target|eval metrics|corr coef with accuracy group|kendall's tau|
|:---:|:---:|:---:|:---:|:---:|:---:|
|Ridge Regression|-|accuracy group|-|0.628|0.467|

Strangely, the weight of prediction which had the best correlation coefficient with accuracy group became 0. But this stacking was so effective. It pushed me up near the gold zone.

# Threshold tuning
A threshold was very important in this competition. At first, I used OptimizedRounder which many kernels used. But I found this function depended on the initial value, and output was likely to fall into a local solution from my experiments. So I used a random search for deciding thresholds. This approach pushed me up about 800th place.  And I thought public and private dataset was very similar because my adversarial validation's AUC was around 0.5. So I selected thresholds that maximize mean-QWK for 100 datasets which were truncated randomly from train dataset.
