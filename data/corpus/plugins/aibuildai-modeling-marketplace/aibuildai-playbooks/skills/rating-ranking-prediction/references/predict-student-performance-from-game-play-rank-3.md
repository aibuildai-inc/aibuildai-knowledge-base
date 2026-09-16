# 3rd Place Solution

Competition: predict-student-performance-from-game-play
Rank: #3
Source: https://www.kaggle.com/c/predict-student-performance-from-game-play/discussion/420235

Thanks a lot to the hosts of the competition and my teammates ( @kingychiu, @tangtunyu, and @yyykrk). I am thrilled that @kingychiu and I will become GM, @tangtunyu is one step closer to becoming a Master, and @yyykrk will get his second gold medal after this competition!

Here We will explain our overall solution, @yyykrk also provided additional explanation of the parts he worked on: https://www.kaggle.com/competitions/predict-student-performance-from-game-play/discussion/420274

# Classification Task Formulation
In this competition, we are asked to predict 18 values for each session. Each session contains 3 level groups. There are multiple ways to model this.
1. 18 binary classifiers
2. 3 Level group classifiers, each one can be
a. A multi-label classifier that predicts all values within a level group
b. A binary classifier that takes “question index” as a feature within a level group
3. 1 classifier that is
a. A multi-label classifier that predicts 18 values within a session
b. A binary classifier that takes “question index” as a feature within a session

For Gradient boosted tree models, method 2b > method 3b > method 1. Method 2a and 3a are ignored because training the multi-label task is a lot slower with Gradient boosted tree models.

For NN models, we focus on the method 2a and 3a, because
- These 2 methods are not well handled by tree models
- Multi-label learning makes more sense, because of the F1 score setting of this competition. (some posts discuss we should not optimize for 1 question).
- Multi-label NN models are faster to train and infer.


# Additional dataset generated from the raw data
We create an additional dataset from the raw data, it contains 11343 complete sessions.
This dataset boosts the CV scores for GBT models  by about +0.001~2, but there is not much effect on the public and private scores, and it has both positive and negative outcomes.
However, it works very well for NN models, we see +0.002 improvement in both CV and public scores.


# Validation
We are using 5-fold GroupKFold on session_id so that there won’t be any seen sessions in the validation set. Also we didn’t include additional data in our validation set.


# Gradient Boosted Tree
Per question classifier is handled by @yyykrk, Per level, and All-in-1 classifier is handled by  @tangtunyu @kingychiu. That’s why there are some inconsistencies in the data preprocessing steps, such as sort by index vs sort by time.

## Per Question Classifiers
We create features for each level group and sorted by index. The features and the sorting methods differ from other models.

```python
# Code in polars
df1 = df.filter(pl.col("level_group") == "0-4")
df2 = df.filter(pl.col("level_group") == "5-12")
df3 = df.filter(pl.col("level_group") == "13-22")

df1 = df1.sort(pl.col("session_id"), pl.col("index"))
df2 = df2.sort(pl.col("session_id"), pl.col("index"))
df3 = df3.sort(pl.col("session_id"), pl.col("index"))
```

##### The number of features:
- Level group 0-4: 1,000 features
- Level group 5-12: 2,000 features
- Level group 13-22: 2,400 features

##### Feature Selection
We try feature selection with out-of-folds but the public scores tend to decrease, so we don’t select features about this model in the final submission.

##### The typical features
- Elapsed time between the previous level group and the current level group.
- Elapsed time and index count between flag events.
- Prediction probabilities for previous questions.
- Sum of the most recent M (M=1,2,...) prediction probabilities.

Flag events are events that must be passed during game progression. We extract them with reference to jo_wilder's source code, game playing, and the log data of users who have got perfect scores. 

##### Single Best Model(5folds XGBoost)
CV: 0.702, Public LB: 0.700, Private LB: 0.701


## Per Level Group Classifiers
In order to allow the level group models to utilize information from previous level groups, we first split the training data by:
```python
# Code in polars
df1 = df.filter(pl.col("level_group") == "0-4")
df2 = df.filter((pl.col("level_group") == "0-4") | (pl.col("level_group") == "5-12"))
df3 = df

df1 = df1.sort(pl.col("session_id"), pl.col("elapsed_time"))
df2 = df2.sort(pl.col("session_id"), pl.col("elapsed_time"))
df3 = df3.sort(pl.col("session_id"), pl.col("elapsed_time"))
```

Feature selection is then applied after feature engineering.

##### Features Engineering

- Room distance and screen distance

```python
    (pl.col("room_coor_x") - pl.col("room_coor_x").shift(1)).over(["session_id"]).pow(2).alias("room_coor_x_dis"),
    (pl.col("room_coor_y") - pl.col("room_coor_y").shift(1)).over(["session_id"]).pow(2).alias("room_coor_y_dis"),    
    (pl.col("screen_coor_x") - pl.col("screen_coor_x").shift(1)).over(["session_id"]).pow(2).alias("screen_coor_x_dis"),
    (pl.col("screen_coor_y") - pl.col("screen_coor_y").shift(1)).over(["session_id"]).pow(2).alias("screen_coor_y_dis"),    

```

- Final scene, checkpoint and answer time

By playing the game manually, we know that students are only taking the quiz at the end of each level. The shorter time they used to finish the session of answering questions, the higher probability that they answered those questions correctly. Captured by features like:

```python
                pl.col("index").filter((pl.col("fqid") == "chap2_finale_c") | (pl.col("event_name") == "checkpoint")).apply(lambda s: s.max() - s.min()).alias("chap2_answer_indexCount"),
                (pl.col("elapsed_time").filter(pl.col("level_group") == "5-12").min() - pl.col("elapsed_time").filter(pl.col("level_group") == "0-4").max()).alias("chap1_answer_time")

```

- Unnecessary moves

Also from the experience of playing the game, we believe that there are many people who have played the game for more than one time. Would be great if we are have some feature to identify these players

```python
unnecessary_data_values = {}
for q in range(23):
    unnecessary_data_values[q] = {}
    for feature_type in ['text', 'fqid', 'text_fqid']:
        unnecessary_data_values[q][feature_type] = []
        unique_values = list(df.filter((pl.col("level") == q))[feature_type].unique())
        
        for val in unique_values:
            if df.filter((pl.col("level") == q) & (pl.col(feature_type) == val))['session_id'].n_unique() < 23000:
                unused_data_values[q][feature_type].append(val)
```

If they are playing for the first time, they likely have many unnecessary moves. Then we calculate the time / actions they have spent of these moves

```python
    for col in ['elapsed_time_diff']:

        aggs.extend([
             *[pl.col(col).filter((pl.col("level") == level) & (pl.col("text").is_in(unused_data_values[level]["text"]))).count().alias(f"level_{level}_unused_text_{col}_counts") for level in level_feature],
             *[pl.col(col).filter((pl.col("level") == level) & (pl.col("fqid").is_in(unused_data_values[level]["fqid"]))).count().alias(f"level_{level}_unused_fqid_{col}_counts") for level in level_feature],
             *[pl.col(col).filter((pl.col("level") == level) & (pl.col("text_fqid").is_in(unused_data_values[level]["text_fqid"]))).count().alias(f"level_{level}_unused_text_fqid_{col}_counts") for level in level_feature],
             *[pl.col(col).filter((pl.col("level") == level) & (pl.col("text").is_in(unused_data_values[level]["text"]))).sum().alias(f"level_{level}_unused_text_{col}_sum") for level in level_feature],
             *[pl.col(col).filter((pl.col("level") == level) & (pl.col("fqid").is_in(unused_data_values[level]["fqid"]))).sum().alias(f"level_{level}_unused_fqid_{col}_sum") for level in level_feature],
             *[pl.col(col).filter((pl.col("level") == level) & (pl.col("text_fqid").is_in(unused_data_values[level]["text_fqid"]))).sum().alias(f"level_{level}_unused_text_fqid_{col}_sum") for level in level_feature],
        ])
    
```

- Time / actions spent on tasks

Another class of feature to filter out experienced  players is to measure how fast they finish the tasks before the quiz in every level group. For example the first task of the game is to find the notebook, our hypothesis is that an experienced player would spend less time and actions to finish it. And they have a higher chance to answer the quiz questions correctly.

Two examples for chapter 1
```python
                pl.col("elapsed_time").filter((pl.col("text") == "Now where did I put my notebook?") | (pl.col("text") == "Found it!")).apply(lambda s: s.max() - s.min()).alias("find_notebook_duration"),
                pl.col("index").filter((pl.col("text") == "Now where did I put my notebook?") | (pl.col("text") == "Found it!")).apply(lambda s: s.max() - s.min()).alias("find_notebook_indexCount"),
                pl.col("elapsed_time").filter((pl.col("text") == "Found it!") | (pl.col("text") == "Let's get started. The Wisconsin Wonders exhibit opens tomorrow!")).apply(lambda s: s.max() - s.min()).alias("go_upstairs_duration"),
                pl.col("index").filter((pl.col("text") == "Found it!") | (pl.col("text") == "Let's get started. The Wisconsin Wonders exhibit opens tomorrow!")).apply(lambda s: s.max() - s.min()).alias("go_upstairs_events")
```

##### Feature Selection
The selection is based on Catboost feature importance over the Catboost feature importance with shuffled labels. (Which is the idea of Null Importances https://www.kaggle.com/code/ogrellier/feature-selection-with-null-importances)
1. Compute Catboost feature importance with the entire training data.
2. Shuffle the training data labels and obtain the importance again for N times.
3. Compute the final importance by the base importance divided by mean random importance.
4. We then use `gp_minimize` to search for the best feature size based on 5-fold cross-validation.
In the end, we have 233, 647, 693 features respectively for each level group.

With Catboost 5-fold CV out of fold F1: 0.7019
With Xgboost 5-fold CV out of fold F1: 0.7021

Then feature engineering is applied to each of the data frames above. And the transformed data frames are used to train our level group models.

##### 18-in-1 Classifiers
To train the 18-questions-in-1 classifier, we further concat the above 3 data frames together to form a large data frame.
```python
# Code in pandas

all_df = pd.concat([
    df1[FEATURES1 + ["q"]],
    df2[FEATURES2 + ["q"]],
    df3[FEATURES3 + ["q"]],
], axis=0)
```
This mega concatenation creates many null values because some features only exist in a particular level group. That’s why when building the features for this 18-in-1 classifier:
First, reuse the feature selection results from the Per Level Group case.
Rerun feature selection again after the mega concatenation

With Catboost 5-fold CV out of fold F1: 0.7002
With Xgboost 5-fold CV out of fold F1: 0.7007


# Neural Network

Model: Transformer + LSTM

The pipeline of our NN is based on this public notebook: https://www.kaggle.com/code/abaojiang/lb-0-694-event-aware-tconv-with-only-4-features

##### Numerical input:
- np.log1p( elapsed_time_diff )
##### Categorical inputs:
- event_comb, room_fqid, page, text_fqid, level


##### Transformer part (3 variants):
- Type A: Conformer like transformer, with last query attention (https://www.kaggle.com/competitions/riiid-test-answer-prediction/discussion/218318)
- Type B: Conformer like transformer, with last query attention
- Type C: Standard transformer



##### Post Transformer LSTM:
- 1 Bidirectional LSTM + 1 LSTM layer


##### Pooling method:
- Concat of sum, std, max, last

##### Training method:
1. As mentioned in the previous section, we train the model with multi-label, and there are two variants:
a. One model per level group
b. Same model for ALL level groups
2. We find that combining models trained with different settings can improve both the CV and public LB.
3. Additional data was used for training, it improves both CV and public LB for NN


### Best NN only ensemble (5 NN with different settings):
- CV: 0.7028, Public LB: 0.701, Private LB: 0.704
- It turns out that NN doesn’t perform very well in Public LB, but does well in Private LB.


# Submission Selection
We selected a submission with the highest LB, a submission with the highest CV, and a submission with a target on a reasonably high CV and a high variety of methods/models. 

Our best-selected sub is an ensemble of 
- One level group Catboost, one 18-in1 Catboost, two 18-in1 Xgboost, and three NN.
- The NNs we selected are Type A per level group, Type B per level group, and Type C ALL level groups. This combination gives good diversity to the final ensemble.
- We ensemble GBT models and NN models on oof data separately with 2 standalone Logistic regression models, then combined them with GBT:NN = 6:4 ratio. 
- The manual weighting in combining GBT and NN results is due to NN not performing well in public LB, so we didn't have enough confidence to give too much weight to our NN models as discussed below.	

Best selected ensemble:
- CV: 0.7046, Public LB: 0.706, Private LB: 0.704


# 0.705 subs that we haven’t picked
We have three 705 private score submissions that are not selected. Our best-selected subs ranked 13th in all of our subs in terms of private score.

Among these 705 private subs:
- Per-question GBT model + Group Level GBT model gives us the 705 private score, but not a high ensemble CV score. 
- Per-level-group GBT + NN models with Logistic regression ensemble gives us the 705 private score, but not a high public score.


# Observations:
1. NN models perform well in CV and private but very poorly in public, while GBT models fit the public so well, It is very strange…
2. Single-question GBT models makes a lower CV ensemble but perform quite ok in both public and private
