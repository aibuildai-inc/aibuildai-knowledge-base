# 9th place solution

Competition: santa-2020
Rank: #9
Source: https://www.kaggle.com/c/santa-2020/discussion/221707

Thank you for hosting this exciting competition ! I'm happy to become a Competition Grandmaster.

# Code

Source code is available here.
https://github.com/toshi-k/kaggle-santa-2020

# Basic Strategy
In this competition, the reward was decided by comparing the threshold and random generated number. It was easy to calculate the probability of getting reward if we knew the thresholds. But the agents can't see the threshold during the game, we had to estimate it.

Like other teams, I also downloaded the history by Kaggle API and created a dataset for supervised learning. We can see the true value of `threshold` at each round in the response of API. So, I used it as the target variable.

In the middle of the competition, I found out that quantile regression is much better than conventional L2 regression. I think it can adjust the balance between **Explore** and **Exploit** by the percentile parameter.

# Features
I made 32 features for training and prediction. I tried some other features, but they didn't work well. Adding unnecessary features sometimes caused time out error, I didn't employ them.

| &nbsp; &nbsp; &nbsp; &nbsp; # &nbsp; &nbsp; &nbsp; &nbsp; | Name | Explanation |
| --- | --- | --- |
| #1 | round | number of round in the game (0-1999)
| #2 | last_opponent_chosen | whether the opponent agent chose this machine in the last step or not
| #3 | second_last_opponent_chosen | whether the opponent agent chose this machine in the second last step or not
| #4 | third_last_opponent_chosen | whether the opponent agent chose this machine in the third last step or not
| #5 | opponent_repeat_twice | whether the opponent agent continued to choose this machine in the last two rounds (#2 x #3)
| #6 | opponent_repeat_three_times | whether the opponent agent continued to choose this machine in the last three rounds (#2 x #3 x #4)
| #7 | num_chosen | how many times the opponent and my agent chose this machine
| #8 | num_chosen_mine | how many times my agent chose this machine
| #9 | num_chosen_opponent | how many time the opponent agent chose this machine (#7 - #8)
| #10 | num_get_reward | how many time my agent got rewards from this machine
| #11 | num_non_reward | how many time my agent didn't get rewarded from this machine
| #12 | rate_mine | ratio of my choices against the total number of choices (#8 / #7)
| #13 | rate_opponent | ratio of opponent choices against the total number of choices (#9 / #7)
| #14 | rate_get_reward | ratio of my rewarded choices against the total number of choices (#10 / #7)
| #15 | empirical_win_rate | posterior expectation of threshold value based on my choices and rewords
| #16 | quantile_10 | 10% point of posterior distribution of threshold based on my choices and rewords
| #17 | quantile_20 | 20% point of posterior distribution of threshold based on my choices and rewords
| #18 | quantile_30 | 30% point of posterior distribution of threshold based on my choices and rewords
| #19 | quantile_40 | 40% point of posterior distribution of threshold based on my choices and rewords
| #20 | quantile_50 | 50% point of posterior distribution of threshold based on my choices and rewords
| #21 | quantile_60 | 60% point of posterior distribution of threshold based on my choices and rewords
| #22 | quantile_70 | 70% point of posterior distribution of threshold based on my choices and rewords
| #23 | quantile_80 | 80% point of posterior distribution of threshold based on my choices and rewords
| #24 | quantile_90 | 90% point of posterior distribution of threshold based on my choices and rewords
| #25 | repeat_head | how many times my agent chose this machine before the opponent agent chose this agent for the first time
| #26 | repeat_tail | how many times my agent chose this machine after the opponent agent chose this agent last time
| #27 | repeat_get_reward_head | how many times my agent got reward from this machine before my agent didn't get rewarded or the opponent agent chose this agent for the first time
| #28 | repeat_get_reward_tail | how many times my agent got reward from this machine after my agent didn't get rewarded or the opponent agent chose this agent last time
| #29 | repeat_non_reward_head | how many times my agent didn't get rewarded from this machine before my agent got reward or the opponent agent chose this agent for the first time
| #30 | repeat_non_reward_tail | how many times my agent didn't get rewarded from this machine after my agent got reward or the opponent agent chose this agent last time
| #31 | opponent_repeat_head | how many times the opponent agent chose this machine before my agent chose this machine for the first time
| #32 | opponent_repeat_tail | how many times the opponent agent chose this machine after my agent chose this machine last time

The feature importance of my best agent is as below.

[importance]

It may be interesting the most important feature was the *round* in the game. I think the agent can change the strategy based on the round, and it was something essential the agent learned from the history.

Some other features related to the opponent agent were also important (*num_chosen_opponent*, *rate_opponent*) . ML based agent can utilize the behavior of the opponent agent which can't be handled by conventional approaches.

# Top Agents
I tried several packages and parameters. It seemed that LightGBM was the best for my settings.

Although some of CatBoost agents achieved similar performance like LightGBM, CatBoost didn't reach the best score of LightGBM. Since all features were numeric variable, it could not utilize the unique feature for categorical variables.

I also tried XGBoost with pairwise loss. It compared the thresholds of machines in the same round. Although I thought it was a natural training setting, the performance of the agents were worse than quantile regression.

NumRound=4000 was almost the maximum number to prevent the time out error. I set  NumRound=3000 or NumRound=3500 for some agents as safer settings. I basically tuned learning rate to control the fitness to the dataset.

| Regressor | Loss | NumRound | LearningRate | LB Score | SubmissionID |
| --- | --- | --- | --- | --- | --- |
| LightBGM | Quantile (0.65)	| 4000 | 0.05 | 1449.4 | [19318812](https://www.kaggle.com/c/santa-2020/submissions?dialog=episodes-submission-19318812)|
| LightBGM | Quantile (0.65)	| 4000 | 0.10 | 1442.1 | [19182047](https://www.kaggle.com/c/santa-2020/submissions?dialog=episodes-submission-19182047)|
| LightBGM | Quantile (0.65)	| 3000 | 0.03 | 1438.8 | [19042049](https://www.kaggle.com/c/santa-2020/submissions?dialog=episodes-submission-19042049)|
| LightBGM | Quantile (0.66) | 3500 | 0.04 | 1433.9 | [19137024](https://www.kaggle.com/c/santa-2020/submissions?dialog=episodes-submission-19137024)|
| CatBoost | Quantile (0.65) | 4000 | 0.05 | 1417.6 | [19153745](https://www.kaggle.com/c/santa-2020/submissions?dialog=episodes-submission-19153745)|
| CatBoost | Quantile (0.67) | 3000 | 0.10 | 1344.5 | [19170829](https://www.kaggle.com/c/santa-2020/submissions?dialog=episodes-submission-19170829)|
| LightGBM | MSE | 4000 | 0.03 | 1313.3 | [19093039](https://www.kaggle.com/c/santa-2020/submissions?dialog=episodes-submission-19093039)|
| XGBoost | Pairwised | 1500 | 0.10 | 1173.5 | [19269952](https://www.kaggle.com/c/santa-2020/submissions?dialog=episodes-submission-19269952)|

# Tips
## Logarithm transformation for probabilistic calculation
The chain of multiplication turns into zero because of the precision of floating point. For the calculation of posterior distribution of threshold, logarithm transformation and renormalization solved this problem.

## LRU Cache
Precise calculation of posterior distribution is time consuming and it can be the cause of time our error. I employed LRU (Least Recently Used) Cache and reused the feature values for the same patterns of the actions and rewards.

# What I should have done (to beat more agents)
When I calculated the winning rate of the agent for features #15-24, I set the probability approximately like `p  = threshold ** decay / 101`. But I missed one important point here. The random generated in each round was integer, I should have rounded this part like `p  = round(threshold ** decay) / 101`.
