# 18th place solution

Competition: riiid-test-answer-prediction
Rank: #18
Source: https://www.kaggle.com/c/riiid-test-answer-prediction/discussion/209597

Congrats to all medal teams and new Grandmasters,Masters,Experts.Thanks to Organizers and kaggle for such a good competition,it shows that kaggle competition is not just a game but also can be a useful machine learning project.

# Team
* At first we have three  teams individually, tomoyo and me, ethan and qyxs , wrb0312.We focus on feature engineering and optimization before wrb0312 joined us,I think we made many good features but neural network dominated this competition.wrb0312 did a great job even he use transformer for the first time.After wrb0312 joined us there are only ten days left,we focus on ensembling our models for inference,and also improved transformer very much.Our team members are from china and japan, it's very interesting to see we use chinese,japanese,english mixed-language to communicate.Greate job everyone!

# Optimization 
* For GBM features, rather than using many dictionary to save features' data, we developed a nubma-based framework to speed up feature engineering process and online calculation. Firstly, the data are sorted by ['user_id', 'timestamp', 'content_id'] and split into different arrays. Then we created features in different array via self-designed rolling function or self-designed cumlative function. Actually, it provides us a very flexible way to create features and test it. In 10m data, the feature engineering process needs only 5 minutes to finish it.

* Some examples are listed as below. 
```
from tqdm import tqdm
from numba import jit,njit
from joblib import Parallel, delayed
from tqdm import tqdm
import gc
from multiprocessing import Process, Manager,Pool
from functools import partial
from numba import prange
import numpy as np
import pandas as pd
from numba import types
from numba.typed import Dict
import functools, time
from numba.typed import List


def timeit(f):
    def wrap(*args, **kwargs):
        time1 = time.time()
        ret = f(*args, **kwargs)
        time2 = time.time()
        print('{:s} function took {:.3f} s'.format(f.__name__, np.round(time2-time1, 2)))

        return ret
    return wrap


def rolling_feat_group(train, col_used):
    a = train[col_used].values
    ind = np.lexsort((a[:,2],a[:,1],a[:,0]))
    a = a[ind]
    g = np.split(a, np.unique(a[:, 0], return_index=True)[1][1:])
    return g, ind, col_used

@jit(nopython = True, fastmath = True)
def rolling_cal(arr, step, window = 5, shift_ = 1):
    m = 2
    arr_ = np.concatenate((np.full((window, ), np.nan), arr))
    ret = np.zeros((arr.shape[0], m))
    beg = window
    for i in step: 
        tmp = arr_[beg-window:beg]
        ret[beg - window:(beg - window + i), 0] = np.nanmean(tmp)
        ret[beg - window:(beg - window + i), 1] = np.nansum(tmp)
        beg += i
    return ret


@jit(nopython = True, fastmath = True)
def rolling_time_cal(arr, window = 5, shift_ = 1):
    m = 1
    arr_ = np.concatenate((np.full((window, ), np.nan), arr))
    ret = np.zeros((arr.shape[0], m))
    for i in range(0,arr.shape[0], 1): 
        tmp = arr_[i:i+window+1]
        ret[i, 0] = np.nanmean(tmp)
    return ret

def rolling_cal_wrap(tmp_g, shift_period):
    m = 2
    tmp_res = []
    step = np.unique(tmp_g[:, 1], return_counts=True)[1]
    for window_size in shift_period:
        tmp = rolling_cal(tmp_g[:, 2], step, window_size)
        tmp_res.append(tmp)
    tmp_res = np.concatenate(tmp_res, axis = 1)
    return tmp_res

def rolling_time_cal_wrap(tmp_g, shift_period):
    m = 2
    tmp_res = []
    for window_size in shift_period:
        tmp = rolling_time_cal(tmp_g[:, 2], window_size)
        tmp_res.append(tmp)
    tmp_res = np.concatenate(tmp_res, axis = 1)
    return tmp_res


def rolling_feat_cal(tmp_g, name_dict, global_period):
    answer_idx = name_dict.index('answered_correctly')
    prior_idx = name_dict.index('prior_question_elapsed_time')
    item_mean_idx = name_dict.index('item_mean')
    task_set_idx = name_dict.index('task_set_distance')
    tmp_res1 = rolling_cal_wrap(tmp_g[:,[0,1, answer_idx]], global_period)
    tmp_res2 = rolling_time_cal_wrap(tmp_g[:,[0,1, prior_idx]], global_period)
    tmp_res3 = rolling_time_cal_wrap(tmp_g[:,[0,1, item_mean_idx]], global_period)
    tmp_res4 = rolling_time_cal_wrap(tmp_g[:,[0,1, task_set_idx]], global_period)
    tmp_res = np.concatenate([tmp_res1, tmp_res2, tmp_res3, tmp_res4], axis = 1)
    return tmp_res
```

* If anyone interested in how to create features via numba-framework, Tomoyo publiced his full GBM pipeline in github([https://github.com/ZiwenYeee/Riiid-numba-framework](url))

# Catboost(LB 0.807)
### summary
* We created 183 features for final catboost model,including some original features,global statistics(item base),cumulative and rolling statistics(user base),tfidf-svd(base on question's user list),word2vec(base on user's question list, wrong and correct tag list ),timedelta from many perspective,last same part groups features.

### gbm benchmark
* We compared lightgbm ,xgboost,catboost,catboost is the best for the training and inference speed,and memory consuming.When train the full data,lightgbm need over 100 hours with my AMD Ryzen ThreadRipper 3970X,xgboost always have out of memory error even using dask with 4 RTX 3090.

### strong features and interesting finding by qyxs
* 1.  the history difficulty statistics features of user who had correct/wrong answers, boost almost 0.003
```
tmp_df = for_question_df.groupby('content_id')['answered_correctly'].agg([['corr_ratio', 'mean']]).reset_index()
tmp_fe = for_question_df[for_question_df['answered_correctly']==0].merge(tmp_df, on='content_id').groupby('user_id')['corr_ratio'].agg(['min', 'max', 'mean', 'std']).reset_index()
for_train = for_train.merge(tmp_fe, on='user_id', how='left')
```

* 2.  focus on the records about the current part of user connect with last same part, generate the features include answer correct ratio, time diff, frequency etc, boost almost 0.002
```
for_question_df['rank_part'] = for_question_df.groupby(['user_id', 'part'])['timestamp'].rank(method='first')
for_question_df['rank_user'] = for_question_df.groupby(['user_id'])['timestamp'].rank(method='first')
for_question_df['rank_diff'] = for_question_df['rank_user'] - for_question_df['rank_part']
for_question_df['part_times'] = for_question_df.groupby(['user_id', 'part'])['rank_diff'].rank(method='dense')
for_question_df['rank_diff'] = for_question_df.groupby(['user_id', 'part'])['rank_diff'].rank(method='dense', ascending=False)

last_part = for_question_df[for_question_df['rank_diff']==1]
part_times = for_question_df.groupby(['user_id', 'part'])['part_times'].agg([['part_times', 'max']]).reset_index()

last_part_df = last_part.groupby(['user_id', 'part'])['answered_correctly'].agg([['last_continue_part_ratio', 'mean'], ['last_continue_part_cnt', 'count']]).reset_index()
last_part_time = last_part.groupby(['user_id', 'part'])['timestamp'].agg([['last_continue_part_time_start', 'min'], ['last_continue_part_time_end', 'max']]).reset_index()
last_part_df = last_part_df.merge(last_part_time, on=['user_id', 'part'], how='left')
last_part_df = last_part_df.merge(part_times, on=['user_id', 'part'], how='left')
last_part_df['part_time_diff'] = last_part_df['last_continue_part_time_end'] - last_part_df['last_continue_part_time_start']
last_part_df['part_time_freq'] = last_part_df['last_continue_part_cnt']/last_part_df['part_time_diff']

for_train = for_train.merge(last_part_df, on=['user_id', 'part'], how='left')
for_train['last_continue_part_time_start'] = for_train['timestamp'] - for_train['last_continue_part_time_start']
for_train['last_continue_part_time_end'] = for_train['timestamp'] - for_train['last_continue_part_time_end']
```

* 3.  the answer correctly ratio of each question under differenct user abilititys (split for 11 bins), boost almost 0.001
```
for_question_df['user_ability'] = for_question_df.groupby('user_id')['answered_correctly'].transform('mean').round(1)
tmp_df = for_question_df.pivot_table(index='content_id', columns='user_ability', values='answered_correctly', aggfunc='mean').reset_index()
tmp_df.columns = ['content_id'] + [f'c_mean_{i}_ratio' for i in range(11)]
for_train = for_train.merge(tmp_df, on='content_id', how='left')
```

* Some interseting points:
 * 1.they would watch lecture after users had wrong answers, so we could generated some features from this. LB is not improved caused by the lectures info in next group maybe.
 *  2.the content_id such as 0-195， 7851-7984 etc, then are all same in one part and continuous with each other，we could build a new bundle to generate features

### strong features and interesting finding by ethan
* 1.  user's behavior in last 1,5,...,60 minutes, 0.001 boost
```
for w in [1, 5, 10, 15, 30, 45, 60]:
    print(w)
    tmp = q_logs[q_logs['timestamp']>=(q_logs['end_time']-w*60*1000)].copy()
    group_df = tmp.groupby(['user_id'])['content_id'].agg([['user_content_nunique_in_last{}mins'.format(w), 'nunique']]).reset_index()
    train = train.merge(group_df, on=['user_id'], how='left')
    group_df = tmp.groupby(['user_id'])['part'].agg([['user_part_nunique_in_last{}mins'.format(w), 'nunique']]).reset_index()
    train = train.merge(group_df, on=['user_id'], how='left')
    group_df = tmp.groupby(['user_id'])['answered_correctly'].agg([['user_correct_raito_in_last{}mins'.format(w), 'mean']]).reset_index()
    train = train.merge(group_df, on=['user_id'], how='left')
 ```
 
* 2. "users' ablility" statistics in each question, seperately by "answered_correctly"(0/1), 0.002 boost
```
cc = q_logs.groupby(['user_id'])['answered_correctly'].agg([['corr_ratio', 'mean']]).reset_index()
gg = q_logs[['user_id', 'content_id', 'answered_correctly']].merge(cc, on=['user_id'], how='left')

group_df1 = gg[gg['answered_correctly']==1].groupby(['content_id'])['corr_ratio'].agg([['question_correct_user_ablility_min', 'min'], 
                                                                                       ['question_correct_user_ablility_max', 'max'], 
                                                                                       ['question_correct_user_ablility_mean', 'mean'], 
                                                                                       ['question_correct_user_ablility_skew', 'skew'],
                                                                                       ['question_correct_user_ablility_med', 'median'],
                                                                                       ['question_correct_user_ablility_std', 'std']]).reset_index()
group_df2 = gg[gg['answered_correctly']==0].groupby(['content_id'])['corr_ratio'].agg([['question_wrong_user_ablility_min','min'], 
                                                                                       ['question_wrong_user_ablility_max','max'],
                                                                                       ['question_wrong_user_ablility_mean','mean'],
                                                                                       ['question_wrong_user_ablility_skew','skew'],
                                                                                       ['question_wrong_user_ablility_med','median'],
                                                                                       ['question_wrong_user_ablility_std','std']]).reset_index()
```
                        
* 3. "lagtime" statistics in each question, seperately by "answered_correctly"(0/1), means the distribution of users' preprare time for answering this question correctly, about 0.001 boost
```
user_task_timestamp = q_logs[['user_id', 'task_container_id', 'timestamp']].drop_duplicates()
user_task_timestamp['lag_time'] = user_task_timestamp['timestamp'] - user_task_timestamp.groupby(['user_id'])['timestamp'].shift(1)
tmp = q_logs[['user_id', 'task_container_id', 'content_id', 'answered_correctly']].merge(user_task_timestamp.drop(['timestamp'], axis=1), on=['user_id', 'task_container_id'], how='left')
group_df = tmp[tmp['answered_correctly']==1].groupby(['content_id'])['lag_time'].agg([['c_lag_time_mean', 'mean'],
                                                                                     ['c_lag_time_std', 'std'],
                                                                                     ['c_lag_time_max', 'max'],
                                                                                     ['c_lag_time_min', 'min'],
                                                                                     ['c_lag_time_median', 'median']]).reset_index()
train = train.merge(group_df, on=['content_id'], how='left')

group_df = tmp[tmp['answered_correctly']==0].groupby(['content_id'])['lag_time'].agg([['w_lag_time_mean', 'mean'],
                                                                                       ['w_lag_time_std', 'std'],
                                                                                       ['w_lag_time_max', 'max'],
                                                                                       ['w_lag_time_min', 'min'],
                                                                                       ['w_lag_time_median', 'median']]).reset_index()
train = train.merge(group_df, on=['content_id'], how='left')
```

### feature list
```
['content_id',
 'prior_question_elapsed_time',
 'prior_question_had_explanation',
 'correct_answer',
 'user_count',
 'user_sum',
 'user_mean',
 'item_count',
 'item_sum',
 'item_mean',
 'answer_ratio_0',
 'answer_ratio_1',
 'answer_ratio_2',
 'bundle_id',
 'part',
 'le_tag',
 'question_correct_user_ablility_mean',
 'question_correct_user_ablility_median',
 'question_wrong_user_ablility_mean',
 'question_wrong_user_ablility_median',
 'word2vec_0',
 'word2vec_1',
 'word2vec_2',
 'word2vec_3',
 'word2vec_4',
 'svd_0',
 'svd_1',
 'svd_2',
 'svd_3',
 'svd_4',
 'tags_w2v_correct_mean_0',
 'tags_w2v_wrong_mean_0',
 'tags_w2v_correct_mean_1',
 'tags_w2v_wrong_mean_1',
 'tags_w2v_correct_mean_2',
 'tags_w2v_wrong_mean_2',
 'tags_w2v_correct_mean_3',
 'tags_w2v_wrong_mean_3',
 'tags_w2v_correct_mean_4',
 'tags_w2v_wrong_mean_4',
 'real_time_wrong_mean',
 'real_time_wrong_median',
 'real_time_correct_mean',
 'real_time_correct_median',
 'task_set_distance_wrong_mean',
 'task_set_distance_wrong_median',
 'task_set_distance_correct_mean',
 'task_set_distance_correct_median',
 'mean_0_ratio',
 'mean_1_ratio',
 'mean_3_ratio',
 'mean_4_ratio',
 'mean_5_ratio',
 'mean_6_ratio',
 'mean_7_ratio',
 'mean_8_ratio',
 'mean_9_ratio',
 'mean_10_ratio',
 'user_d1',
 'user_d2',
 'task_set_distance',
 'user_diff_mean',
 'user_diff_std',
 'user_diff_min',
 'user_diff_max',
 'task_set_item_mean',
 'task_set_item_min',
 'task_set_item_max',
 'task_set_distance2',
 'task_distance_shift',
 'task_set_distance_diff',
 'task_distance_diff_shift',
 'container_mean_1',
 'container_mean_5',
 'container_std_5',
 'container_mean_10',
 'container_std_10',
 'container_mean_20',
 'container_std_20',
 'container_mean_30',
 'container_std_30',
 'container_mean_40',
 'container_std_40',
 'prior_question_elapsed_time_mean_1',
 'prior_question_elapsed_time_mean_5',
 'prior_question_elapsed_time_mean_10',
 'prior_question_elapsed_time_mean_20',
 'prior_question_elapsed_time_mean_30',
 'prior_question_elapsed_time_mean_40',
 'item_mean_mean_30',
 'item_mean_mean_40',
 'task_set_distance_mean_1',
 'task_set_distance_mean_5',
 'task_set_distance_mean_10',
 'task_set_distance_mean_20',
 'task_set_distance_mean_30',
 'begin_time_diff',
 'end_time_diff',
 'part_time_diff_mean',
 'part_session_mean',
 'part_session_sum',
 'part_session_count',
 'full_group0_item_mean_mean',
 'full_group0_item_mean_median',
 'full_group0_task_set_distance_median',
 'full_group0_timestamp_mean',
 'full_group0_timestamp_median',
 'full_group1_item_mean_mean',
 'full_group1_item_mean_median',
 'full_group1_task_set_distance_median',
 'full_group1_timestamp_median',
 'part_sum',
 'part_count',
 'part_mean',
 'part_sum_global_ratio',
 'part_sum_1',
 'part_sum_5',
 'part_mean_5',
 'part_sum_10',
 'part_mean_10',
 'cum_answer0_mean_item_mean',
 'cum_answer0_median_item_mean',
 'cum_answer0_median_task_set_distance',
 'cum_answer1_mean_item_mean',
 'cum_answer1_median_item_mean',
 'cum_answer1_mean_task_set_distance',
 'cum_answer1_median_task_set_distance',
 'cum_answer0_time_diff',
 'cum_answer1_time_diff',
 'global_task_set_shift1',
 'global_task_set_shift2',
 'global_task_set_shift4',
 'global_task_set_shift5',
 'cum_answer0_mean_wrong_time_diff',
 'cum_answer0_median_wrong_time_diff',
 'cum_answer1_mean_right_time_diff',
 'content_correct_mean',
 'content_correct_sum',
 'content_correct_count',
 'hard_answer0_time',
 'hard_answer1_time',
 'full_bundle_item_mean_mean',
 'full_bundle_item_mean_median',
 'full_bundle_task_set_distance_mean',
 'full_bundle_task_set_distance_median',
 'full_bundle_timestamp_mean',
 'full_bundle_timestamp_median',
 'bundle_sum',
 'bundle_mean',
 'bundle_count',
 'user_trend_mean',
 'user_trend_median',
 'user_trend_roll_user_ans_sum',
 'user_trend_roll_user_ans_mean',
 'user_trend_roll_user_ans_count',
 'user_trend_roll_item_ans_mean',
 'user_trend_roll_item_ans_count',
 'div_ratio1',
 'div_ratio2',
 'div_ratio3',
 'new_Feat0',
 'new_Feat1',
 'new_Feat2',
 'new_Feat3',
 'part_time_wrong_div',
 'part_time_right_div',
 'diff_lag_median_div',
 'diff_item_median_div',
 'diff_time_median_div',
 'diff_item_mean_div',
 'diff_task_set_mean_div',
 'diff_timestamp_mean_div',
 'last_20_frequent_answer',
 'last_20_frequent_answer_count',
 'last_20_frequent_answer_mean',
 'last_20_frequent_answer_sum',
 'last_user_same_answer_tf',
 'last_item_same_answer_tf',
 'last_right_time_diff',
 'last_wrong_time_diff',
 'last_5_part_time_div',
 'last_10_part_time_div',
 'last_20_part_time_div']
```
# Transformer(LB 0.808)
* Used questions only. Lectures did't improve our validation score.
*  In training window size 800 (the bigger,the better), in inference window size 300(the bigger,the better,but time consuming).
* Optimizer : Adam  with lr = 8e-4, beta1 = 0.9, beta2 = 0.999 with warmup steps to 4000.
  * Without warmup steps, didn't converge.
* Number of layers = 4, dimension of the model = 256, dimension of the FFN = 2048.
* Batch size=80
  * With smaller size,  didn't converge.
* Dropout = 0
* Postion encoding : Axial Positional Embedding
  * https://arxiv.org/abs/1912.12180
  * https://github.com/lucidrains/axial-positional-embedding
  * This encoding improved score significantly.
* Augmentation : Mixup-Transformer
  * https://arxiv.org/abs/2010.02394

* Inputs
  * question_id
  * part
  * prior_question_elapsed_time / 1000
  * lagtime
      * log1p((timestamp_t - timestamp_(t-1)) /1000 / 60)
      * This feature improved the score significantly
  * answered_correctly
  * GBT feats (imortance top N)
* model image


# Ensemble(LB 0.812)
* Finally we use one catboost and two transformers for ensemble due to the inference time limitation.Unfortunately our inference notebook has bug although we improved our model about 0.0006 at last day.
