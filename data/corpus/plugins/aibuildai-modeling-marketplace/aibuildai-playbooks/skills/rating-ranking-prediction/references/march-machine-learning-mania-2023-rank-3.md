# 3th Place Solution

Competition: march-machine-learning-mania-2023
Rank: #3
Source: https://www.kaggle.com/c/march-machine-learning-mania-2023/discussion/401641

**Private Leaderboard Score:** 0.17525
**Private Leaderboard Place:** 3th
**[Notebook link](https://www.kaggle.com/tihonby/march-madness-3th-place-solution/)**

#### Introduction
Hello Kaggle Members! 
I want to thank the organizers for hosting this contest, Kaggle for providing a platform for the contest. Thanks to the colleagues at Kaggle for the public codes that I learned a lot from. Thanks to @raddar and @verracodeguacas for their codes that I used in my work. My congratulations to the finalists and winners of this wonderful competition!
#### Background
I am ML Engineer, 3 years of professional experience, this is my first time participating in this competition, I used to predict the results of other sports.
#### Datasets
In the final, I only used data from NCAATourneyDetailedResults, NCAATourneySeeds, RegularSeasonDetailedResults. I experimented with MasseyOrdinals data, but in the end I abandoned this data, as I did not find a clear increase. I think that it is necessary to return to them, since I have not studied them enough.
#### Features
Features that were used in training the final model:'Score',  'FGM',  'FGA',  'FGM3',  'FGA3',  'OR',  'Ast',  'TO',  'Stl',  'PF',  'seed', and derivatives from them. When testing, I excluded other features, as they did not give an increase in the quality of predictions. With an increase in the number of features over 70, the quality of forecasts began to deteriorate.
Added data extraction function for the last 3 seasons. The function retrieves the value of the feature in the current season and in the previous two. Assigns weights in descending order. The current season carries more weight and contributes more. Calculates and writes the average value to a new function. This feature allows you to smooth out outliers when a team performs differently this season than in previous years. In some cases, this can improve the forecast by 0.001-0.003. The function is little studied and is applied to the parameters 'Score', 'FGA', 'FGA3', 'quality', 'seed'.
#### Scaling
I used Normalizer, MinMaxScaler, StandardScaler. The best results are from Normalizer, which was used in the final.
#### Models
XGBoost, ElasticNet, LogisticRegression have been tested.XGBoost has better performance, final settings: xgb.cv, repeat_cv=4, max_depth=3, folds=KFold, n_splits=5.
When tested with repeat_cv = 7-10 gave an improvement of 0.003-0.005 over the multi-season average. But if the team that the algorithm was sure of had lost, the score could have been worse, which happened in this tournament. For this tournament, the result for repeat_cv=7 was worse than for 4. I checked the options with a forecast of the probability of winning 1 or 0, or a forecast by points, I did not find a significant difference.
#### Results processing
I did not post-process the final results of the model prediction, the predictions were in the original version.Later I rechecked for 1-3, 14-16, but this did not noticeably improve the results.I do not welcome manual adjustment, because in this way you can reduce all predictions to a manual forecast based on rating and personal preferences. Although I think that it can improve the forecast and add emotions during the competition.
The search for optimal parameters was carried out by training in all seasons, except for the last one, which was used for validation. The training was held separately for men's and women's teams. Prediction results are better for women's matches. Approximately 0.145-0.151 versus 0.187-0.192. I think that the men's teams have more equal strength, and it is more difficult to predict the outcome of the match in advance. The average result for the seasons is about 0.172-0.173, the difference for individual seasons is 0.169-0.177. I expected my result in the competition to be in the range of 0.174 - 0.176, although I thought that there would be those who could predict better than 0.169.
The 2 most successful options were selected for sending, the results of the forecasts: 0.17525 (3rd) and 0.1763 (6th) places. Some predictions for strong teams turned out to be incorrect, but others had the same predictions for this game, so the place in the rankings did not change much. The results obtained are very pleasing.
#### Source
@raddar: [Here's the original R from 2018](https://github.com/fakyras/ncaa_women_2018/blob/master/win_ncaa.R). The guidelines of @raddar - Darius Barušauskas, [youtube presentation](https://www.youtube.com/watch?v=KmhGNc7gcCM&t=18s&ab_channel=Kaggle)
@verracodeguacas Python notebooks from 2023: 1: [men](https://www.kaggle.com/code/verracodeguacas/xgboost-ken-pom-data-sagarin-ranks), 2: [women](https://www.kaggle.com/code/verracodeguacas/xgboost-ncaa-women-s-side), 3: [mixer](https://www.kaggle.com/code/verracodeguacas/easy-mixer-men-women)

##### Thanks Kaggle! Good luck to everyone in the new season!
