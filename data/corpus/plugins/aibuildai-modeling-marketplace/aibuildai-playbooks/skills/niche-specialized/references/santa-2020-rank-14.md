# 14th Place Solution (Looking for teammates)

Competition: santa-2020
Rank: #14
Source: https://www.kaggle.com/c/santa-2020/discussion/221529

We'd like to thank the Kaggle team for hosting the Santa 2020 competition during a tough year. In this competition, I had the pleasure of teaming up with [Alex](https://www.kaggle.com/lihuajing), [Nischay Dhankhar](https://www.kaggle.com/nischaydnk), [Priyanshu Chaudhary](https://www.kaggle.com/chaudharypriyanshu), and [Per von Soosten
 (PS)](https://www.kaggle.com/soosten). This is a summary of our team's solution which placed 14th on the leaderboard. Our [best agent](https://github.com/soosten/candy-canes/blob/main/agent.py) is relatively simple, with only about 100 lines of code ([github repo](https://github.com/soosten/candy-canes)).

TL;DR: We relied on a Bayesian Bandit approach with a few heuristics.

### Local Simulations
We used a pool of opponents to evaluate our agents. Running a large number of matches against these opponents gave us a rough idea of agent performance, but improvements in local win rate did not always translate well to the leaderboard. Our pool of opponents included some of our own agents and a few agents from public notebooks:
* opponent_agent.py and submission.py from [pull_vegas_slot_machines add weaken rate continue5](https://www.kaggle.com/a763337092/pull-vegas-slot-machines-add-weaken-rate-continue5)
* thompson.py from [Santa 2020: Thompson Sampling](https://www.kaggle.com/xhlulu/santa-2020-thompson-sampling)
* bayesian_ucb.py from [Santa 2020: UCB and Bayesian UCB Starter](https://www.kaggle.com/xhlulu/santa-2020-ucb-and-bayesian-ucb-starter)
* submission.py from [Simple multi-armed bandit](https://www.kaggle.com/ilialar/simple-multi-armed-bandit)

Thank you to [Lindada焱焱焱](https://www.kaggle.com/a763337092), [xhlulu](https://www.kaggle.com/xhlulu) and [Ilia Larchenko](https://www.kaggle.com/ilialar) for making these notebooks public. 

### Bayesian approach
The basic logic of our final submission is to maintain a set of distributions reflecting what we know about the current reward probability of each arm. At the outset, these distributions are initialized to a uniform prior. Every time we get a result from our previous pull, we perform a Bayesian update on the corresponding distribution reflecting the result of our previous pull. This is a classical idea for dealing with multi-armed bandits.

### Choosing the next arm
We decide on which new arm to pull based on a simplified variant of the Bayesian UCB algorithm. We get an optimistic estimate of the reward probability of each arm by adding the mean and the standard deviation of the corresponding distribution and then choose an arm with a maximal estimate. An important part of the game is to understand when the opponent has discovered an arm with a high reward probability before the opponent has exploited that arm enough to dramatically decrease the probability. We tried to incorporate this information by applying some temporary Bayesian updates to the distribution before computing the mean and standard deviation comprising the estimate. More concretely:
* If the opponent pulled an arm more than twice in the last 10 turns, we assumed all of those pulls resulted in a reward.
* If the opponent pulled an arm exactly once in the last 100 turn, we assumed that pull did not result in a reward.

Our local simulations indicated that the optimal strategy never pulls on an arm with a reward probability less than 0.22. Therefore, we stopped applying temporary updates to arms for which the true estimate (without any temporary updates) was less than 0.25. This ensured that we did not follow the opponent to arms which we already knew were bad.

### Decaying thresholds
In principle, the distributions should also reflect the decay in the reward probability after each arm is pulled. However, we noticed that our agents that did not use the opponent's actions to decay their estimates performed surprisingly well. One explanation for this is that not decaying encourages the agent to exploit good machines that the opponent has found, this is especially important early in the game. On the other hand, having correct estimates of the thresholds is important near the end of the game. We attempted to get the best of both worlds by averaging these two estimates (with/without opponent decay) with a weight that favors non-decayed estimates early in the game and decayed estimates later in the game.

### Learning from opponent actions
Finally, some opponent actions are so indicative of the result their pulls that we conservatively applied some permanent Bayesian updates to the distribution based on them. We hardcoded the logic needed to learn from these opponent actions. This can work well against agents who are greedy with respect to their threshold estimates, but these hardcoded rules are easy to counter by being deceptive. Here are the rules that we used
* If the opponent repeats a first-time action, assume the first time is a success.
* If the opponent doesn't pull a lever for a long time after pulling it for the first time, it is probably because the first time was a failure.
