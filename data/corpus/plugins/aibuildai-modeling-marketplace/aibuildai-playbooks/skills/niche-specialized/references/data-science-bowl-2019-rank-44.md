# 44 place writeup(Catboost ranking with eventdata)

Competition: data-science-bowl-2019
Rank: #44
Source: https://www.kaggle.com/c/data-science-bowl-2019/discussion/127304

First, thanks Kaggle and Booz Allen Hamilton for such a great competition.
It was an interesting problem with lots of challenges and I learned a lot.

Here is my solution and observations:

**Loss function and framework**
I use ranking loss(`PairLogitPairwise:max_pairs=1000000`) with CatBoost, depth 6 or 7, training on GPU.
The other hyperparameters are default.
I train the model 5 fold and then blend all 5 models with CatBoost `sum_model `to produce the average prediction.
So I obtain a quasi single model solution.
I try to predict `accuracy_group`. My attempts to predict `accuracy`, or `num_correct `and `num_incorrect `as targets didn't work well.

**Validation**
5-fold truncated cross validation where I perform truncation 5 times for each fold and average the score.

**Threshold selection**
After the model is blended I predict the whole training set and optimize the threshold to maximize the Kappa. 
I do it 5 times and then take the median value for each sample.

**Features generation**
I've generated about 1000 features. Among them are:
1. Overall accumulated counters of event_codes and event_ids.
2. Various accumulated accuracy statistics.
3. Timestamp month and hour
4. Linear extrapolation of accuracy.
5. Features extracted from event data:
    a. Overall sum and mean value for each key that has numeric value except coordinates.
    b. The same statistics groupped by event title.
    
Here is the few top features by SHAP importatance to illustrate the idea:
- `lastAssessmentTitle`
- `misses_mean`
- `Bird Measurer (Assessment)_stage_number_mean`
- `accuracy_mean`
- `4070_count`
- `Sandcastle Builder (Activity)_total_duration_mean`
- `IsAssessmentAttemptSuccessfull_Chest Sorter (Assessment)`
- `Clip_count`
- `6bf9e3e1_count`

**Feature selection**
Features of group 5b (like `Bird Measurer (Assessment)_stage_number_mean`) lead to a heavy overfitting for training set. To mitigate that two approaches work:
1. Select top 150-200 features by shap.
2. Drop features using truncated adversarial validation untill ROC AUC becomes ~0.5. That leaves 863 features.

**Submission selection**
I've submitted the most stable blend of 3 models that vary by the selected features and produce 0.555-0.56 at public LB. That produced .552 private and 44 place.
I have few single model and blend submissions for .553 and .554, so my final submission was quite close to optimal and I get a fair Silver.
