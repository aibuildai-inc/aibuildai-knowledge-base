# [1st Place Solution] - Our Betting Strategy

Competition: ubiquant-market-prediction
Rank: #1
Source: https://www.kaggle.com/c/ubiquant-market-prediction/discussion/338220

Thanks to Ubiquant and everyone involved for hosting the contest.
And all the competitors who were with us on the six-month journey worked hard.

In particular, the Alphas provided by Ubiquant showed relatively stable and high correlation with the target in the ever-changing financial market.
It was a valuable and enjoyable experience to utilize such high-quality data. Thanks again.

Although we were the winners, this is the result of many Kaggler's discussions and public notebooks.
Also, I think we were lucky enough that the market conditions made our model smile.

Nevertheless, we want to share the process we went through to increase our chances of getting this luck.


I. Summary
>A. Used Models: LGBM, TABNET
>B. Feature Engineering: 300 + 100
>C. Data Sampling: (train.csv + supplemental_train.csv)[2400000:]
>D. Cross Validation for FE and Parameter Tuning: PurgedGroupTimeSeries, TimeSerieseSplit
>E. Cross Validation for Training : KFold


II. Detailed description

A. Used Models : LGBM, TABNET
1. Sorry if you were expecting a special model, this time it's LGBM and TABNET. :D
2. LGBM is a powerful model whose performance has been proven in many competitions. It was also the most stable (especially the consistency of CV and LB) and excellent in the experiment on the competition data. (at least for me)
3. In addition, candidate models to be ensembled in LGBM were found and tested. Among them, TABNET was selected with the best ensemble effect while being relatively stable.
4. Some Custom MLP models were also candidates, and there were models with a significant ensemble effect even on the LB basis, but as a result, they were excluded because they were not stable in CV.
5. For loss_fn, rmse and mse are used, respectively, and Pearson Corr. is commonly used for eval_metric.
6. The ensemble method : Average of (LGBM x 5 Folds) + (TABNET x 5 Folds)

B. Feature Engineering: 300 + 100

1. First, to reduce or remove the influence of unnecessary features, we investigated the importance and corr. of each feature, but did not find any meaningful numerical evidence, and then conversely, we contemplated the features to be added.
2. The above 300 means built-in features, and the added 100 features showed consistent and significant improvement in CV and LB Score (CV: 0.141 -> 0.154, LB: 0.141 -> 0.149 based on LGBM single model) .
3. Although we only used 100 additional features, this was the largest compromise within the limits allowed by Kernel Resources (especially RAM), and I guess that there would be an additional score increase if more useful features were included.
4. The added 100 is calculated in the following way.
[ The average value at each time_id for the top 100 features by obtaining and sorting the corr. of 300 features and each target ]
[ Code ]
>features = [f'f_{i}' for i in range(300)]

>corr = train_df[features[:] + ['target']].corr()['target'].reset_index()
>corr['target'] = abs(corr['target'])
>corr.sort_values('target', ascending = False, inplace = True)
>best_corr = corr.iloc[3:103, 0].to_list()

>time_id_mean_features = []
>for col in tqdm(best_corr):
>&nbsp;&nbsp; mapper = train_df.groupby(['time_id'])[col].mean().to_dict()
>&nbsp;&nbsp; train_df[f'time_id_{col}'] = train_df['time_id'].map(mapper)
>&nbsp;&nbsp; train_df[f'time_id_{col}'] = train_df[f'time_id_{col}'].astype(np.float16)
>&nbsp;&nbsp; time_id_mean_features.append(f'time_id_{col}')

>features += time_id_mean_features


C. Data Sampling: (train.csv + supplemental_train.csv)[2400000:]
1. The above means the concatenation of train.csv and supplementa_train.csv, meaning that we used the last 2400k rows of it.
2. The reason for this sampling was related to FE of 2, of course, the added features caused a huge memory increase, and we had to trade off [more features] VS [more data].
3. For this, various probabilistic measures were performed on Score gains on PurgedGroup and TimeSeriesSplit CV, and we finally decided that additional features have a probabilistic advantage in score improvement.
4. After that, as a result of testing in memory (RAM 13GB), the Data Row was stable up to approximately 2500k, but in order to pursue more stability, an additional 100k was dropped. :(

D. Cross Validation for FE and Parameter Tuning : PurgedGroupTimeSeries, TimeSerieseSplit
1. As already described above, various CVs were used to measure the performance of FE, and the most effective FE was selected from all of these CVs.
2. This course also includes Hyper Parameter Tuning.

E. Cross Validation for Training : KFold, GroupKFold
1. There was another trivial strategy for this choice, which is a self-test data set, which looks like this:
training set: (time_id >= 0) and (time_id <= 1000)
test set: (time_id >= 1001) and (time_id <= 1202)
2. There were various training methods, but as a result of testing in the above environment, [limited training KFold] was selected as the method that showed the most stable and excellent results.
3. To elaborate a bit, this has the advantage of being able to include more data in a variety of ways, while at the same time risking overfitting for future references.
4. So, to reduce the risk of overfitting, we used an early stop for validation and a method of limiting the number of training (num_boost_round or epoch) to a certain value or less.


P.S.
 - Second Submission?
I didn't mention it above, but yes, I also want to keep the second submission short.
Our second submission takes a very different direction from the above.
First, supplementa_train.csv was not included, and the above model, mlp, and catboost were included, and there were a little more about 150 features, and it was considerably overfitted with unlimited training (of course, there is an early stop for validation). However, this submission resulted in a silver medal score(0.115796). In this part, I think there was an effect from the exclusion of supplemental data or excessive CV and LB overfitting.
Then, in time series data training, I think that the method such as early stop at the time when the CV score decreases to a few percent or less is also a factor to consider, but it is only a guess.


Many Thanks.


=============================================
Answering for this guys > @agenlu @hdynamics @ygygyv 
How to make a feature from test? > Take the average of each feature in the test set by referring to the list of features stored in best_corr in the code above.
like this,
[Code]
>for col in best_corr:
>&nbsp;&nbsp;test_df['time_id'] = test_df['row_id'].str[0:4].astype(np.int64)
>&nbsp;&nbsp;mapper = test_df.groupby(['time_id'])[col].mean().to_dict()
>&nbsp;&nbsp;test_df[f'time_id_{col}'] = test_df['time_id'].map(mapper)
