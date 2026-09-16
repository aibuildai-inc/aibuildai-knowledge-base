# 9th place solution : 6 Transformers + 2 LightGBMs

Competition: riiid-test-answer-prediction
Rank: #9
Source: https://www.kaggle.com/c/riiid-test-answer-prediction/discussion/210354

First of all, thanks to the hosts for a great competition! This was one of the toughest competitions I have ever entered, but well worth the effort.

Prediction of our team (tito @its7171 and nyanp @nyanpn)consists of following models.

- tito's transformer: LB 0.813
- nyanp's SAINT+ transformer: LB 0.808
- nyanp's LightGBM: LB 0.806

Simple blending of these models scored LB 0.814 / private 0.816.

## Pipeline

We converted the entire train.csv data into hdf5, and loaded only the user_id that appeared during inference into a np array (97% RAM savings compared to hold entire training data in RAM). We estimate that the overhead due to I/O in hdf is ~45 minutes. This some overhead allowed us to combine tito's large transformer with nyanp's feature engineering pipeline.



## Transformer (tito, LB 0.814)
This is transformer model with encoder only, based on @claverru's [nice kernel](https://www.kaggle.com/claverru/demystifying-transformers-let-s-make-it-public)

### summary
* only trained and predicted for answered_correctly of last question of the sequence.
* all features are concatenated (only position encoding is added).
* used lectures, in timestamp order as it is
* Window size 300-600
* batch size 1000
* drop_out 0
* n_encoder_layers 3-5
* augmentation to replace content_ids with dummy ids at a certain rate
* kept only one question in the last task to avoid leaks

### features
embedded or dense was decided by CV.
* (embedded) content id
* (embedded) part id
* (embedded) same task question size
* (dense) answered_correctly
* (dense) had_explanation
* (dense) elapsed time
* (dense) lag time
* (dense) diff of timestamp from the last question

### combining models
To avoid the overhead of calling model.predict() multiple times for ensemble, I made a combined model that links four models.

```python
inputs = tf.keras.Input(shape=(input_shape, n_features))
out1 = model1(inputs[:,-window_size1:,:])
out2 = model2(inputs[:,-window_size2:,:])
out3 = model3(inputs[:,-window_size3:,:])
out4 = model4(inputs[:,-window_size4:,:])
combo_model = tf.keras.Model(inputs, [out1,out2,out3,out4])
```


## SAINT+ (nyanp, LB 0.808)
- d_model = 256
- window_size = 200
- n_layers = 3
- attention dropout = 0.03
- question, part, lag are embed to encoder
- response, elapsed time, has_explanation are embed to decoder

To prevent leakage, in addition to upper triangular attention mask, I masked the loss in questions other than the beginning of each task_container_id. Questions with the same task_container_id were shuffled in each batch during training, and the loss weights were adjusted to reduce the effect of masks. This mask improved LB by 0.0003.

(Note: I believe that indirect leaks still exist, but I've spent 80% of my time on the LightGBM implementation and data pipeline, so I couldn't improve it any further)

Other than that, there is nothing special about this NN. It scored .806 in single, .808 by averaging 2 models.

## LightGBM (nyanp, LB 0.806)
LightGBM models are trained on 264 features. To speed up inference, I fixed the number of trees to 3000 and ensembled two models with different seeds (This is better than single large LGBM in terms of both speed and accuracy). By compiling this model with [treelite](https://github.com/dmlc/treelite), inference time became 3x faster (~10ms/batch, ~10min in total).

### features
I mapped prior_question_* rows with their respective rows by following code and utilized them in some features.

```python
df['elapsed_time'] = df.groupby('user_id')['prior_question_elapsed_time'].shift(-1)
df['elapsed_time'] = df.groupby(['user_id', 'timestamp'])['question_elapsed_time'].transform('last')

df['has_explanation'] = df.groupby('user_id')['prior_question_had_explanation'].shift(-1)
df['has_explanation'] = df.groupby(['user_id', 'timestamp'])['has_explanation'].transform('last')
```

Here is the list of my features. There is no magic here; no single feature boost CV more than 0.0002. I repeated feature engineering based on well-known techniques and a little bit of domain knowledge.

#### question features
- count encoding
- target encoding
- number of tags
- one-hot encoding of tag (top-10 frequent tags)
- SVD, LDA, item2vec using user_id x content_id matrix
- LDA, item2vec using user_id x content_id matrix (filtered by answered_correctly == 0)
    - Typical word2vec model is trained on next-word prediction task. By constructing word2vec model over incorrectly answered questions, 
the latent vectors extracted from the model can be used to capture which incorrect question are likely to co-occur with each other.
- 10%, 20%, 50%, 80% elapsed time of all users response, correct response, wrong response
- SAINT embedding vector + PCA

#### user features
- avg/median/max/std elapsed_time
- avg/median/max/std elapsed_time by part
- avg has_explanation flag
- avg has_explanation flag by part
- nunique of question, part, lecture
- cumcount / timestamp
- avg answered_correctly with recent 10/30/100/300 questions, recent 10min/7 days
- avg answered_correctly by part, question, bundle, order of response, question difficulty, question difficulty x part
    - question difficulty: discretize the avg answered_correctly of all users for each question into 10 levels
- cumcount by part, question, bundle, question difficulty, question difficulty x part
- lag from 1/2/3/4 step before
- correctness, lag, elapsed_time, has_explanation in the same question last time
- tag-level aggregation features
    - calculate tag-level feature for each user x tag, then aggregate them by min/avg/max
    - cumcount of wrong answer, cumcount of correct answer, avg target, lag
- (estimated elapsed time) - (avg/median/min/max elapsed time within same part)
    - estimated elapsed time = (timestamp - prev timestamp) / (# of questions within the bundle)
- (estimated elapsed time) - (10%, 20%, 50%, 80% elapsed time of all users correct/wrong response)
    - Because TOEIC part1-4 questions are usually answered after listening to the conversation, the correct answers tend to be concentrated immediately after the conversation ends
- task_container_id - previous task_container_id
    - I'm not sure why this worked. There might be a difference in the correct rate if people answered from different devices than usual (multi-user?).
- part of last lecture
- lag from last lecture
- whether the question contains the same tag as the last lecture
- whether the part is the same as the previous question
- median lag - median elapsed_time
- part of the first problem the user solved
- lag - median lag
- inner-product of user-{correct|incorrect}-question-vector and question-vector
    - user-correct-question-vector: average of LDA vectors for each question that the user answered correctly.
- rank of lag compared to the same user's past lag (filtered by answered_correctly == 0, 1 respectively)


### feature calculation
Instead of updating the user dictionary, I calculate user features from scratch for each bacth.

The numpy array of historical data was loaded from the hdf storage and then split into questions and lectures, which were then wrapped in a pandas-like API and passed to their respective feature functions.

```python
@feature('lag1.user_lecture')
def lag1_user_lecture(df: RiiidData, pool: DataPool) -> np.ndarray:
    """
    time elapsed from last lectures for each user

    :param df: data in current batch
    :param pool: cached data storage
    """
    lag1 = {}
    for u, t in set(zip(df.questions['user_id'], df.questions['timestamp'])):
        past_lect = pool.users[u].lectures  # history of lecture
        if len(past_lect) > 0:
            lag1[u] = t - past_lect['timestamp'][-1]

    return np.array(list(map(lambda x: lag1.get(x, np.nan), df.questions['user_id'])), dtype=np.float32)
```

For training, I use the same functions as inference time. This way, there were almost no restrictions on feature creation, and I did not have to worry about bugs of train-test difference.

The feature functions were frequently benchmarked by a dedicated benchmark script, and functions with high overhead were optimized by various ways (numba, bottleneck and various algorithm improvement) .

The question features were precomputed and made into a global numpy array of shape (13523, *) and merged into the feature data using fancy index.

```python
@feature(tuple(f'question_item2vec_{i}' for i in range(20)))
def question_item2vec(df: RiiidData, _: DataPool):
    """
    question embedding vector using item2vec
    """
    qid = df.questions['content_id']

    ret = QUESTION_ITEM2VEC[qid, :]  # mere fancy index, faster than pd.merge

    return ret
```


## Other ideas
- TTA on SAINT+ by np.roll (it did improve SAINT+ by 0.001+, but we couldn't include it because of submission timeout)
- linear blending based on public LB labels (timeout, too)

## Feedback on Time-Series API Competition
Although the competition was well-designed, our team still found that the time-series API allowed us to obtain private score information with 1-bit probing even after the known vulnerability was fixed.

```python
env = riiideducation.make_env()
iter_test = env.iter_test()

ground_truth = []
prediction = []
threshold = 0.810

for idx, (test_df, sample_prediction_df) in enumerate(iter_test):
    ground_truth.extend(list(test_df['answered_correctly'].values))

    predicted = model.predict(...)

    prediction.extend(list(predicted))

    if len(prediction) >= 2500000:
        private_auc = roc_auc_score(prediction[500000:], ground_truth[500000:])
        if private_auc < threshold:
            raise RuntimeError()

# private auc of this submission is guaranteed to exceed .810 if submission is succeeded
```

By using this probing, we can select the final submission with the highest private score, or determine the best
 ensemble weight for private score by using the hill-climbing method.

We contacted the Kaggle Team and asked them if this is legal, and they said it was a "low signal matter". We still think this is a gray-area and decided not to use this probing.

We think it wasn't critical in this competition, but if there is a future competition with the same format but without AUC metrics, the hack with the "magic coefficient" could improve the ranking significantly. If a competition with the same format is held on Kaggle in the future, we suggest the Kaggle team to fix this problem (e.g. put a dummy value in the private label during the competition).
