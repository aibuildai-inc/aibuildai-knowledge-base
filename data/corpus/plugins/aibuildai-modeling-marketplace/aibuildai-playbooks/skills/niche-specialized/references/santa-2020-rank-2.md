# [2nd place] My approach - LightGBM quantile regression

Competition: santa-2020
Rank: #2
Source: https://www.kaggle.com/c/santa-2020/discussion/221030

Good luck to everyone with the last a few days of the evaluation period!

# Overview
I submitted mainly 4 types of models:

| agent | no. of submissions  | best score as of 21st Feb|
| --- | --- | --- |
| 1. UCB-like rule based agent | 43 | 951.9 |
| 2. Reinforcement learning |20 | 997.9 |
| 3. LightGBM regression & greedy| 105 | 1442.1 |
| 4. LightGBM quantile regression | 30 | 1521.1  |


This figure is quoted from nagiss's excellent leaderboard analysis [notebook][2] and shows estimated scores of each agents along the submission dates. 

The last two LightGBM models (3, 4) are described in this note. The models are uploaded [here][1].

# Model
LightGBM regressor was used to predict original threshold of each bandit, and the action is decided by argmax(predicted thresholds * decay).
Agent 3 uses normal LGBM regression to predict original thresholds. Agent 4 is a combination of two different quantile regressions: 60 percentiles before 750 steps and 65 percentiles after that. These parameters are chosen by local evaluations to maximize chosen threshold difference against agent 3 (see Evaluation section below).

## Training data
20k-30k episodes are used in the training. Episode extraction criteria was changed a few times, but basically 100 episodes per team per day, team rank >= 30 and agent score >= 1100. 

I don't share the script here because the used api list_episodes_for_team is no longer available. I kept updating my dataset through the competition until the api was disabled at the last week of competition.

## Features
The main 15 features used in the LightGBM are as follows.

	1. Current step
	2. Number of times my agent chose the bandit
	3. Number of times my agent chose the bandit, corrected by 1/decay factor at the time of choice
	4. Number of times my agent obtained reward from the bandit
	5. Number of times my agent obtained reward from the bandit, corrected by 1/decay factor at the time of choice
	6. Hit rate (4. / 2.)
	7. Adjusted hit rate (5. / 2.)
	8. Number of times opponent chose the bandit
	9. Number of times opponent chose the bandit, corrected by 1/decay factor at the time of choice
	10. Number of steps passed since my agent chose the bandit
	11. Number of steps passed since opponent chose the bandit
	12. Rank of 8. in the 100 bandits
	13. Rank of 9. in the 100 bandits
	14. Rank of 11. in the 100 bandits
	15. Threshold decay factor (0.97 ** (2. + 6.))

As you can see in the uploaded scripts, I've added some more rank features, however, it doesn't affect the leaderboard score so much.

## Metrics and hyper parameters
Agent 3. uses simple LightGBM regression and parameters are followings.
`
params = {'metric': 'rmse',
          'feature_fraction': 1.0,
          'num_leaves': 253,
          'bagging_fraction': 0.8488399520000339, 'bagging_freq': 2,
          'lambda_l1': 0.00012247089654570252, 'lambda_l2': 3.9068674768413283e-05,
          'min_child_samples': 100}
`
Agent 4. used LightGBM quantile regression with two different alpha. The other hyperparameters are the same as above.
`
params1 = {'objective': 'quantile', 'alpha': 0.60}
`
`
params2 = {'objective': 'quantile', 'alpha': 0.65}
`
[Optuna LightGBMTuner][3] was used for hyper parameter tuning. I feel it is time-efficient and really easy to use for a beginner like me.

# Evaluation
To evaluate the model performance, I used win% and also cumulative difference of chosen threshold. For example, the figure below shows cumulative differences in threshold and reward between my model 3. vs 4., averaged over 500 games. Even with smaller number of games, the threshold diff was fairly more stable than rewards.



# Some general thoughts:
- I thought "exploitation" is much more important than "exploration" in this game. The goal is not to get as many reward as possible, but to get more reward than opponent's. Even if there are undiscovered better bandits, it doesn't matter as long as the opponent doesn't pull it neither.
- I suspected that there are many teams using threshold prediction & greedy and need to beat this kind of agents. Slight improvements in prediction accuracy didn't affect win% so much in my local evaluations, so I tried quantile regression inspired by UCB to change the agent behavior a bit to deceive opponents keeping higher expected reward.
- Luck is really an important factor in this game. I tried to submit agents as many as possible - this strategy has some risk to expose my agent's behavior but number of submitted agents might be more important.

# What didn't work (for me)
- Reinforcement learning. It was quite difficult for me to train models with this volatile reward/outcomes. Really looking forward to seeing someone's RL agent implementation if any.
- Quantile regression with alpha < 0.5. I thought exploration is not needed in this game so tried to choose more "confident" bandits, but such agents were weaker than normal regression agent.

# Acknowledgement
I started writing my agents after reading iehn's [notebook][4]. Through my attempts to improve it, I learned how important it is to exploit information from opponent's actions, and some of the features like adjusted hit rate estimation and rank features.
lebroschar's [notebook][5] was a great baseline to build greedy decision tree model. Combining this idea with what I learned from rule-based agents, my agents started to get into gold medal zone.

I often had a look at nagiss's [learderboard analysis][6] to see if current ranking is due to sheer luck or not.. It encouraged me and gave some confidence through the competition.

[1]: https://www.kaggle.com/kibuna/santa2020-lightgbm-quantile-agent
[2]:  https://www.kaggle.com/nagiss/santa2020-stable-rating-estimation-leaderboards
[3]: https://medium.com/optuna/lightgbm-tuner-new-optuna-integration-for-hyperparameter-optimization-8b7095e99258 
[4]: https://www.kaggle.com/iehnrtnc/santa2020
[5]: https://www.kaggle.com/lebroschar/1000-greedy-decision-tree-model
[6]: https://www.kaggle.com/nagiss/santa2020-leaderboard-analysis
