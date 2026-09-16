# 6th solution

Competition: mlb-player-digital-engagement-forecasting
Rank: #6
Source: https://www.kaggle.com/c/mlb-player-digital-engagement-forecasting/discussion/271890

First of all, I also would like to thank this competition host and all competitors. It has been about two years since I got my first silver medal in the APTOS competition, and I'm finally a Kaggle master. I'm very pleased now. I'm still new, but I'll continue to learn from Kaggle with a sincere attitude.
 I am also happy that I could get a gold medal in a competition related to my favorite sport what is baseball. Baseball may be a minor sport on the global stage, but it's very popular in Japan, and I watch NPB (Nippon Professional Baseball Organization) games every week. I wish this fascinating and strategic would become more popular around the world!

# To member ([tea](https://www.kaggle.com/tea1013) & [sqrt4kaido](https://www.kaggle.com/nomorevotch))
 I've been participated in competitions as a solo, and this MLB competition was my first experience participating as a team. I learned a lot of things when I participated competitions by solo, but I learned even more this time. I could get a gold medal thank to your a lot of ideas.

# tea's part
## Models
- LGBM  lag / no lag
- CatBoost  lag / no lag
Use **optuna** to optimize hyper-parameters.
## Train & Valid
- Use 2018-01-01 ~ 2021-05-31 data for train, and 2021-06-01 ~ 2021-07-17 for validation
- During the test period, I use 2018-01-01 ~ 2021-06-30 data for train
- Limit data to in-season data only
## Features mainly effective
- target lags (for 45 days)
- player target statistics (mean, median, max, min, var, skew, kurt)
    - Use the respective statistics for April, May, and June 2021.
    - Also use the respective statistics for game day and no game day.
- team target statistics (mean, median, max, min, var, skew, kurt)
    - Use statistics for June 2021 only
- daysSinceLastGame / Roster
- days from the beginning of the year / month
- day of week
- years from the debut year
- age
- position
- player status
- playerBoxScores features

<br />

# sqrt4kaido part
## Models
- LGBM only no lag
1. The first seed, only the players in test are used for training.
2. Second seed, only the players in test are used for training.
3. The first seed. All the players are used for training.
Use **optuna** to optimize hyper-parameters.
Before update train.csv, the best score for my single model was 1.3146.
## Train & Valid
- training phase
Use 2018-01-01 ~ 2021-03-31 data for train, and 2021-04-01 ~ 2021-04-30 for validation
- after update train.csv
Use 2018-01-01 ~ 2021-05-31 data for train, and 2021-06-01 ~ 2021-06-30 for validation
I did not use the data for the first half of July because, unlike the other months, the values were very small.
- Limit data to in-season data only <- important
## Features mainly effective
- player target statistics (mean, median, max, min, var, skew, kurt)
Use the respective statistics for June 2021. This feature is leaked to the valid data, but we used it because it was also valid for public testing.
- team target statistics (mean, median, max, min, var, skew, kurt)
same as above
- season info
pre/post season? regular season? alltardate? etc.
- award flag
- daysSinceLastGame / Roster <- important
- day of week
- years from the debut year
- age
- position
- player status
- playerBoxScores features
## did not work
- Create separate models for the data with and without matches.
After update train.csv, it stopped working for some reason.
- Undo the scale and learn
This time the target is scaled from 0 to 100, but there was a [discussion](https://www.kaggle.com/c/mlb-player-digital-engagement-forecasting/discussion/253940) that it seems to be max-scaled for each day.
So I tried training with the scale back, and the CV results improved. However, when we actually applied it to the test data, the score worsened because the wrong scaling was applied when the max prediction was shifted, so all the predictions for the day were shifted.
I also tried scaling to the average of the same month in the previous year, but this did not improve the accuracy.

<br />

# Makabe's part
## Models
- I used three models.
  - Simple NN (add lags)
  - Light GBM (add lags)
  - Light GBM (none lags)

- The reason to use three type models is the following.
  - To ensemble is useful making robust models.
  - Regarding the lag feature, <strong>which is the key to this competition</strong>, I noticed the following fact in the middle of the competition.
    - (If we can get the correct target information as a lag early in the test period.)
      -  Light GBM (add lags) > Simple NN (add lags) > Light GBM (none lags)
    - (If we have to use the target information predicted by model as lag late in the test period.)
      -  Light GBM (none lags) > Simple NN (add lags) > Light GBM (add lags)
    - From the above, I can see that lag feature is very effective if it contains the correct values. However, because of its effectiveness, there is also a possibility of overfitting. This tendency was especially for LGB.
    - I decided to control the blend ratio of the three models according to the number of days in test period to balance lag features (effectiveness & overfitting).
    - The method of controlling the size of lag features depending on test time period was pointed out by [other Kagglers](https://www.kaggle.com/c/mlb-player-digital-engagement-forecasting/discussion/256620) too. <string>It was very effective method</strong> and we could this knowledge for our entire team ensemble too.

## Training / Validation
- I used 5 folds.
  1. < 20200801(training) & 2020/08/01 - 2020/09/30(validation)
  2. < 20200901(training) & 2020/09/01 - 2020/10/31(validation)
  3. < 20210401(training) & 2021/04/01 - 2021/05/31(validation)
  4. < 20210501(training) & 2021/05/01 - 2021/06/30(validation)
  5. < 20210601(training) & 2021/06/01 - 2021/07/xx(validation)

- I decided on period with the following two points in mind.
  - Season with irregular schedule due to Covid-19 (2020).
  - The trend of target feature is very different between the first and second half of the season.

- However, I hadn't been able to determine the period above based on a deep consideration of this impact. I think there is room for reconsideration.

## Features
 The following is a list of feature what is helped improve accuracy. (There were many useful features than the below, but most of them were already in other Kaggler's notebooks, so I omit them.)
- stats of target feature
  - I labeled the data by period as follows, and I added stats that is the one before as features.
    - 2018.3 & 2018.4 (=label1), 2018.5 & 2018.6 (=label2), ...
    - I thought that by doing this, and I could use stats of target until September 2021. However, I knew after the competition, test period was until August 31, so I didn't need to use the stats of target two months earlier.
  - I used mean, median, distribution,.. and more.
  - I calculated the stats of targets by defense position, roster status, and team, respectively.
    - There was difference of targets by roster status. (The difference was especially noticeable in target1.)

- number of days what have elapsed since the last time player participated in game.
  - This ideas was from [@sqrt4kaido](https://www.kaggle.com/nomorevotch). This variable was based on the fact that the engagement of players who haven't played for a long time tends to decline over time, and contributed to improving accuracy.

- score from the day before yesterday
  - I thought the success of a game is continuous, and player's popularity is not necessarily based only on the previous day's performance, but past success is a factor.
  - [Team AutoMLB seems to this point](https://www.kaggle.com/c/mlb-player-digital-engagement-forecasting/discussion/266596), and incorporated player scores into the model for a considerable period of time. I probably should have included more time periods than just the day before yesterday.

- A lot of stats
  - I used various features based on [Sabermetric](https://sabr.org/sabermetrics). (K/BB, RC, OPS, FIP.. and more)
  - This is especially for pitchers, I thought the ratio divided by the number of innings pitched or the number of pitches thrown tends to be a better representation of the player than the absolute number. (It's no surprise that starting pitchers strike out more than relief pitchers.) For this reason, I also incorporate many indicators divided by the number of innings or throw of pitches(K9, HR9.. and more).

<br />

# Doing by team
- In order to ensemble easily, it was necessary to improve the reproducibility and reusability of each member’s source code. Therefore, we unified the specifications and developed the prediction classes with common methods.
- In this competition, it was very important to finsh the process correctly, because the Time Series API is very complicated. Therefore, we created a LocalTestClass to reproduce the Time Series API in local environment.
- By using this class, we could verify the accuracy in a larger period.
- We controlled the ensemble ratio for the fact that the effectiveness of lag feature weakens over time.

[[MLB-sub.png]](https://postimg.cc/gwwHCDJm)
