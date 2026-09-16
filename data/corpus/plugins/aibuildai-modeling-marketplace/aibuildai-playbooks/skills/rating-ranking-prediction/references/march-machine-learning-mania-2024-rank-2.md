# 2nd Place Solution

Competition: march-machine-learning-mania-2024
Rank: #2
Source: https://www.kaggle.com/c/march-machine-learning-mania-2024/discussion/492761

# 2024 March Madness Mania Competition

## [2nd Place Submission](https://www.kaggle.com/competitions/march-machine-learning-mania-2024/leaderboard)

[GITHUB Link](https://github.com/JackLich10/kaggle_2024)

```
Private Leaderboard Score: 0.05437
Private Leaderboard Place: 2nd
```

### Background

I am a 2022 graduate of Duke University with a degree in Data Science. I currently work for a sports analytics company in Washington, D.C.. This is my third year entering this competition. I entered again because I think it's super fun to forecast the results of such crazy basketball tournaments. Last year, I finished 7th in the competition, before that 100th. My men's model was first written 3 years ago for fun and I have updated both my men's and women's models this year to use the same methodologies.

### Methodology

All coding was done in R. The basic methodology behind both of my models is to:

1) generate team ratings via ridge (`glmnet`)/mixed effects (`lme4`) regressions predicting point difference, offense/defense efficiency, offense/defense pace, etc. (i.e. what is team A's impact on offense efficiency, score difference, etc. after controlling for opponent, home court, etc.?). I also fit a 'matchup adjustment' mixed effects model, which tries to predict if a team will play up or down to the competition.

2) use team ratings from 1) to predict team level offensive efficiency (points/possession) and pace (possessions) for each game (via XGBoost)

3) use team ratings from 1) and team level predictions from 2) to predict game level score difference (via XGBoost)

The most important features to predict the team level efficiencies and the game level score differences are the various team ratings from 1). Each of these models are trained after each day of each season. As such, it takes multiple days for the entirety of this modeling framework to run for the first time. A simple GLM converts the predicted game level score difference to predicted win probability. I predicted each possible matchup's win probability (similar to the submission format of previous competitions) and then conducted 100k simulations of each tournament.

### Code

This [repository](https://github.com/JackLich10/kaggle_2024) has my men's model and women's code in their respective folders, `mens/` and `womens/`. I also have a common `models/` folder with each team/game level model that can be run for either men's or women's, depending on command line options. Within each of `mens/` and `womens/`, the `run_all.sh` shell script specifies the order in which the files need to be run for the entire modeling framework to work. The `simulate.R` script takes the predictions and simulates the tournament, combining into the final submission file.

*NOTE*: My men's model uses public data via my own [gamezoneR](https://jacklich10.github.io/gamezoneR/index.html) R package and from the public [hoopR](https://github.com/sportsdataverse/hoopR/) R package. My women's model uses data from the public [wehoop](https://github.com/sportsdataverse/wehoop/) R package. Shout out sportsdataverse and ESPN for public data feeds!

### Manual Overrides

In my first submission, I submitted bracket simulations from my raw unaltered win probabilities. In my second simulation, I altered the probabilities such that the UConn men’s team would win with probability 100% for every possible game - this is the submission that won 2nd place. My unaltered submission would have finished 16th.

### Thoughts on Format

Firstly, I want to thank @kaito510, @danielcljc, @jacobyjaeger, and others who were in many discussions, comments, etc. with great arguments against the original scoring metric (average bracket score). I also want to thank the Kaggle organizers for listening to community feedback and having the humility to alter the scoring metric to something much better!

However, while the brier scoring on round probabilities is certainly a much better system than the original average bracket score, I am *not* a fan of this scoring.

In my opinion, the game theory behind this type of scoring incentivizes gambling more than previous years - specifically to predict the champion with 100% probability. I have yet to see other winning submissions, but I am willing to bet that many over-rid champions as I did (either SC or UConn or both). It all has to do with how round probabilities/brackets enforce a dependency between games that was irrelevant in previous competitions.

There is an interesting distinction here where under old rules (each game is independently scored based on win probability), if someone correctly predicts the champion they would *easily* win the competition - that is, by a margin even larger than the leaders this year! The reward for predicting the champion correctly under the old rules is higher in absolute terms. However, the game theory under the old rules is definitely not to predict the champion, this is why nobody used that strategy in older competitions! For example, the 0/1 flip is a great strategy under the old rules, but under the new rules it is pretty insignificant. Under a metric that independently scores each game, at any point someone can “stop” their override and incur no penalty. Under the new rules, the penalty for “stopping” an override is harsh because of having strong early round positions on the team(s) in question. In fact, the only way this penalty is not incurred with the new scoring is if you predict a team all the way through as the winner - the conclusion I made when submitting UConn to win with 100% probability. 

In other words, I believe the game theory under the old rules is certainly not to predict the winner, it is much too risky. Under the new rules, predicting the winner is the **only way** to actually achieve a significant boost above the rest of the field! I believe that if this metric is used again, next year’s competition will be filled with even more people who submit a gamble with a 100% predicted champion.

### My Preferred Scoring Metric

In my opinion, the best scoring metric is clearly MAE on point differential (not RMSE, every point should be worth the same). Here are the reasons why:

- Reduces variance and rewards strong machine learning models more so than other metrics!
    - A flip of score difference from -1 to +1 is almost insignificant in relation to a flip in binary win loss 0/1.
- No clear gambling/risk taking strategies! 
    - It’s trivially much easier to flip a win probability from 20% to 100% if you want to bet on a given team, how someone does that with MAE on point difference is much less clear.
- You know what to root for! 
    - This is the *only* metric where regardless of knowing other people’s submissions, you will always know what to root for - the point difference to be as close to your prediction as possible! (It’s way more fun to know what to root for during this competition).

I would also be in favor of reducing submissions from 2 to 1, eliminating gambling incentives further.

In general, whatever the format ends up being next year, I hope the organizers take the time to think about the *game theory* that the scoring metric induces when people are making submissions optimized to win top prizes/medals. I am very thankful that Kaggle continues to host this top tier march madness competition!
