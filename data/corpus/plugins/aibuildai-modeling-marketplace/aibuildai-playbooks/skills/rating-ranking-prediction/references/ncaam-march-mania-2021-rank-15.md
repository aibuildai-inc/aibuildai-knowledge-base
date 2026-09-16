# 15th place short writeup

Competition: ncaam-march-mania-2021
Rank: #15
Source: https://www.kaggle.com/c/ncaam-march-mania-2021/discussion/231032

### Approach
- Each of the 2 submissions is based on a completely different single model
- No postprocessing / gambles

#### Model1: catboost
The first model is a simple catboost model, using several features including difference of elo rating.

The key is to use each team's elo rating **at the time of each game**, rather than at the end of the regular season.





Taking the UCLA-Gonzaga as an example, we can see from the bracket that their game will be played in the final four. This means that **UCLA and Gonzaga must have won four tournament games before their match**.


Of course, we don't know which team UCLA will actually beat (except in the first round), but we can make a good estimate of the elo rating at the time of the game by running a simulation.


In this way, the team after the upset win is regarded by the model to have a better chance of winning, because their elo rating will increase significantly (and apparently that's [true](https://www.ncaa.com/news/basketball-men/article/2021-02-12/heres-how-teams-do-after-they-pull-ncaa-tournament-upset)).

#### Model2: Transformer

The second model is a Single Query Transformer based on the [Riiid 1st place solution](https://www.kaggle.com/c/riiid-test-answer-prediction/discussion/218318), trained on the results of regular season games.

I'm not going to go into details since the results were not very good, but I'm attaching the code for the model so you can take a look if you are interested.

https://drive.google.com/file/d/1tClZol39KMrcQFQJ3Nmw7uhWBxJg5S1R/view?usp=sharing

### Result
The results, including back-testing in past tournaments, are as follows. It seems that this year I was simply lucky ;)

| year | catboost | Transformer |
| --- | --- | --- |
| 2021 | **0.57154 (15th)** | 0.65211 (482th) |
| 2019 | **0.4813** (237th) | 0.5106 (484th) |
| 2018 | 0.6110 (449th) | **0.5881** (165th) |
| 2017 | 0.4936 (102th) | **0.4893** (80th) |
| 2016 | **0.5403** (42th) | 0.5553 (116th) |
