# 5th Place Solution : My First Competition and First Gold

Competition: mlb-player-digital-engagement-forecasting
Rank: #5
Source: https://www.kaggle.com/c/mlb-player-digital-engagement-forecasting/discussion/271345

Hi all

I would like to thank kaggle and the organizers for such a good competition. 
I also thank  my teammate([@Hyper-Positive-Yancy](https://www.kaggle.com/shinnyayoshida)), he did some EDA and tuned NN.


# Models Used 
Our final ensemble consisted of
- Lightgbm X 8
- CATBOOST X 4
- ANN X1

# Data usage period
- **in-season sampling**
I only use in-season data.
Even if I extracted out-of-season data from our data, we could not confirm any deterioration in accuracy.
But, In 2019 data , I eliminated a lot of data. In this year, The retirement match of the great Ichiro Suzuki was held in Japan. He didnt play well in the retirement game, but he had a high engagement.
I considered this data outlier and deleted it. From this fact, I Concluded, Special matches(like retirement match) should be removed from the data

# Feature engineering
- **gamesStartedPitching lag feature**
As I mentioned [here](https://www.kaggle.com/c/mlb-player-digital-engagement-forecasting/discussion/261357#1446144)
I used "gamesStartedPitching lag feature". Starting pitcher sometimes pitch (one time per 4~5days) but they get high target1~4. I made sub model of predicting "this pitcher will pitch tomorrow". If this pitcher will pitch tomorrow, he will get high target.
"This pitcher will pitch tomorrow" is easy to predict, because MLB pitchers have rotation (almost of them pitch one time per 4~5days)
when I use this feature with "target-lag-feature", my public LB score became better. 

- **tomorrow game exists or not**
In this competition, we predict "tomorrow" engagement, but stats of player is today stats. Even if the players did well today, we could see that the engagement value would be low if there was no match tomorrow.
So, Whether or not there will be a match tomorrow will be an important factor in predicting engagement

- **target-lag-feature**
As I mentioned before, this feature is effective when used with "gamesStartedPitching lag feature". Therefore, I thought it would be very meaningful to use the target-lag-feature of  3 to 7 days.
But, Second half of the evaluation period, I have to use predicted value of target data.
It may be one of the factors that worsen our second half model performance.

- **batter contribution**
- **pitching contribution**
In order to evaluate all athletes fairly, we used the evaluation index for athletes.
I mainly use two evaluation index " batter contribution" and "pitching contribution"
I found  " batter contribution" [here] (http://maddog31.xyz/baseball-web/contribution_degree/contribution_degree2/) sorry this page is written in japanese.
I found "pitching game score"  [here](https://www.mlb.com/glossary/advanced-stats/game-score)
As a problem, even with the same position pitcher, there are types of relief and starting pitcher, and the pitching game score of the relief tends to be low. So the probability that the player will throw as starting pitcher has also been added as a feature.



# validation scheme
For training and validation, I used data from each in-season.
After the train data was updated, I changed validation scheme.

Training : 4/1/2018 to 6/30/2021
Validation : 7/1/2021 to 7/31/2021
