# 10th place Solution

Competition: data-science-bowl-2019
Rank: #10
Source: https://www.kaggle.com/c/data-science-bowl-2019/discussion/127332

First, thanks kaggle team for a exciting competition, and congratulations to all winning teams and good result teams.
I joined this competition solely, so it was hard but very interesting competition.

1 year ago, I have experienced very big shake down(2th =&gt; about 1500th) at Microsoft Malware Competition.
https://www.kaggle.com/c/microsoft-malware-prediction/discussion/83950

From this experience, I made effort to validation strategy and public/private analysis.
As a result I got my first Gold Medal by shake up.

Here is my solution. (I am sorry for my poor English.)

## Results

10th(solo Gold) / 3523

## Model

- LightGBM x 6 average
  - CV seed snd some hyper parameters is changed per model
  - After averaging regression value, transform to integer accuracy_group by threshold.

## Validation

- Stratified Group KFold 10fold
- All Validation score is calculated by truncated validation.
  - random sample assessment each installation_id
- Each fold, I use 51 truncated validation set.
  - 1 set is used for early stopping
  - 50 set is used for validation score by averaging qwk.

## Public vs Private

I think public dataset is not good distribution for validation because there are only 1000 records.
I calculated by 1000 times the average of train oof prediction which is truncated and randomly sampling 1000 rows.
The histgram is as follows.



From this histgram, It seems that the public dataset is rare case.

Therefore I trust CV (ignore LB) and use threshold of CV best (explain detail in next).

## QWK threshold

Finaly I used constant threshold [1.04, 1.76, 2.18].
This threshold is calculated by OptimizedRounder's threshold average in 500 truncated oof validation.

I tryed many methods, but I believed maximum threshold in local CV prediction is most reliable.

Some public kernels decided threshold by target distribution.
In my experiment, the method is good for public LB than other methods, but I think this method is overfitting to Public LB because the distribution is not equal to the truncated target distribution and not best distribution for QWK.


## Feature

I made 3000~5000 features overall, but I think there are no magic features.
(Finaly I used about 300 features.)

Good features for me is as follows.

- Normalized Accuracy feature
  - I normalized accuracy features because the difficulty of assessments and games are different per title.
    - (Accuracy - Accuracy_mean_per_title) / Feature_std_per_title
  - accuracy features means accuracy_group, n_true_attempts/all_attempts, correct/event_num and corret/(correct+false) etc...
- Feature per title
  - I make feature per title because the difference of level in a game is difficult to find common columns in event data.
    - Ex : target_distances length in Air Show
  - However it spends a lot of time, so I make only abount 10 titles(game, assessment) and gave up...
- Relative feature
  - Ex: event_code: 4020_count / 4070_count, last_accuracy / all_accuracy_mean
 

## Feature Selection

To evaluate features effect in truncated validation, I use LGB feature importance by truncated training data.
In each fold, I make 50 truncated datasets, and change dataset per 5 iteration by using lightgbm init_model params.

I use top 300 features (the number is feeling).

## Others

- LightGBM parameter feature fraction =&gt; 1.0
  - Feature fraction change (0.8 =&gt; 1.0) make improvement my CV about 0.005
  - I think that the model should use assessment_title for every tree because the title has big effect on the target value, and the role of other features is change by the title feature.
   (It is hypothesis. I don't know it is correct)
- model per game session
  - Ubove Transformer model, I make lightgbm model per game session (predict next assessment result).
  - The model is not used for the main model, but it is useful for find good feature in game eventdata speedy.
- Use test dataset for training
  - I don't know it made improvement.

## Not Work

- NN regressor (MLP)
  - Though NN sometimes has good score, but not stable.
  - I have no time to tuning.
- NN EventCode Transformer
  - I regard one session as one sentence, and event codes as words.
  - Prediction next assessment per session, and use it as feature
  - A little improvement but consuming long time, so I do not use it.
- Word2Vec Feature
  - Similar to Transformer, I regard one session as one sentense.
  - No improvement.
- Predicting normalized accuracy group
  - No improvement.
- Training redidual error per title
  - No improvement
