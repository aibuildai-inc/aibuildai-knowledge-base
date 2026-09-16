# 11th place solution

Competition: mlb-player-digital-engagement-forecasting
Rank: #11
Source: https://www.kaggle.com/c/mlb-player-digital-engagement-forecasting/discussion/264396

**[update]**
In the end, I got 11th place and I was able to stay within the gold medal range. 
I'd like to thank the people running this competition and all the people I competed with! Thank you very much.


a first week of RE-RUN is over. 
I was lucky enough to have a successful rerun.
I would like to thank the admin and all the participants for organizing this event.
There is still a month to go before it ends, but before I forget, I'll describe my solution.
(I'm not very good at English, so I basically translate Japanese at DeepL. I apologize if you have trouble reading.)
<br>


***


####  Difficult points of this competition
1. The target is scaled from 0 to 100 on a daily basis.
1. There is a difference in the information available to pitchers and batters.(The roles of the pitcher and batter are fundamentally different.)

<br>

#### My approach to the above points
- **Use only in-season data for training**
When I checked the daily median values, I found that there was a considerable difference in the median values between the on-season and off-season.
I thought that on-season engagement would be determined by performance and success during the season, while off-season engagement would be determined by media exposure and social networking regardless of the season(and we can't use this kind of data).
In this competition, we predicts the engagement during the season, so I judged that data from the off-season would be noisy and did not use it for training and validation.

- **Create features that represent what happened on any given day and the game**
I thought that daily median engagement scores might differ depending on what happened in the game as a whole and what happened in other games on that day.
Therefore, I created the features related to the actions that occurred throughout the day.
  - Number of games played that day, total runs scored, home runs, hits, etc.
  - The average pct of both teams in each game (whether there was an exciting game on the same day between teams with high winning percentages)
  - The total twitter followers of the teams playing in the game (home and away) as a percentage of the total followers of all teams (whether the game was played between teams that are popular on social networking sites)
  - The total number of twitter followers of the players who played in the game that day as a percentage of the total number of followers of all players (how influential are the players who played that day on SNS)
  - How long has the season elapsed (what percentage of the schedule has been completed)?<br>
In addition to the above, I added the player's performance.<br><br>

  - Daily performance
  - Season performance
  - Previous season performance (last season's final performance)

- **Built three different models**
one for pitchers, one for batters, and one according to no play for the 2021 season.
The features used in each model is slightly different from each other. 
The features for any given day or game described above are the same for all models, 
but in the pitcher's model, the only feature used in the player's performance was the performance related to pitching. in the batter's model, only the performance related to batting was used. 
For the model of a player without a season of play, we built the model without using any features related to performance.<br>
The reason for this, as mentioned earlier, is that pitchers and batters have distinctly different roles. For example, even if a pitcher gets three hits, his reputation will not be high if he is scored on by his crucial pitches.
In addition, for players who did not play in the season, we could not obtain their performance in the current season at all and had no material to judge them, so we built a separate model with fewer features.(no use performance features) 
<br>
As a side note, I don't use target-lag-feature.
When I was testing the three models separately, the accuracy of the pitcher and batter models decreased when using past targets, so I decided not to use variables related to past targets this competition.


<br>

#### validation scheme
For training and validation,  I used data from each season.
In addition, I did not use the data for 2020 because the number of games was small and we thought that there might be abnormal movements due to the corona disaster.
All models are made with lightgbm.

<br>

##### Before the train data was updated (LB:1.3104(Previous leaderboard))
 - Training : 4/1/2019 to 4/30/2019
 - Validation : 5/1/2019 to 5/30/2019 → check accuracy and obtained parameters in this period
 - Re-training: re-train with parameters obtained in the above period on data from 4/1/2021 to 4/30/2021

<br>

#####  After the train data is updated

After the train data was updated, I tested both.
a. whether August of the previous regular season could be predicted well (2019. except for 2020) 
b. whether the last month of the same season could be predicted well.
 The parameters obtained for each were used to make each prediction, and the final submission was weighted  a and b.

- Validation a: 
  - Training : 4/1/2019 to 7/30/2019 
  - Validation : 8/1/2019 to 8/31/2019
  - check accuracy and obtaining parameters in this period

- Validation b: 
  - Training : 4/1/2021 to 6/17/2021
  - Validation : 6/18/2021 to 7/17/2021
  - check accuracy and obtaining parameters in this period

- Re-training: 
   Re-train on data from 2021/4/1 to 7/17 with parameters obtained in the above period, respectively.

<br>

Final sub 1: 
Predicted value of the model made with the parameters obtained in verification a × 0.7 +
Predicted value of the model made with the parameters obtained in verification b × 0.3

Final sub 2:
Predicted value of the model created with the parameters obtained in verification a × 0.3 + Predicted value of the model created with the parameters obtained in verification b × 0.7

<br>
***

##### postscript
Since my solution does not use lag-target, I believe that it is inferior to models that use lag-target for training and inference if we only look at the results of the first week.
However, I imagine that the accuracy of these models using lag-targets may gradually decrease in future re-runs, and I expect that there will be a few shake-up/down.
(But first, I need to get the next rerun running correctly...)

Thank you for reading this far!
