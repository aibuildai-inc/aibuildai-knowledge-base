# 29th public - Am I the only one that approach this with regression?

Competition: nfl-big-data-bowl-2020
Rank: #26
Source: https://www.kaggle.com/c/nfl-big-data-bowl-2020/discussion/119382

First and foremost, my background is not from computer science, got no degree or master or phd in that. A lot of what I learn is from self learning. So my solution is probably most different :D

I was interested with RL since the publication of deepmind's go reinforcement learning algorithm so I had been following their progress and tried to replicate their model. Reading the crps description which says  predict a cumulative probability distribution for the yardage gained or lost.  I somehow was reminded of deepmind's C51 and QR-DQN distributional architecture which maps a state to output where the output is a distribution of outcome which a learning agent had seen in training data.
C51        : https://deepmind.com/research/publications/distributional-perspective-reinforcement-learning
QR-DQN: https://arxiv.org/pdf/1710.10044.pdf


Unlike the initial RL algorithm which maps state to a single expected value of an action, distributional RL algorithm gives a range of values per action which then enabled their agent to more accurately describe the distribution of outcome and enables the choice of risk seeking or risk avoidance behaviour. For example, given a few choice of action the least risky is argmax(mean-std) of the action, while risk seeking is argmax(mean + std). That was the idea of building RL with distributional output estimation.

So back to this competition, since it is simply a state(X) -output(Y) mapping. I use the distributional output to train NN for the different range of yards distribution that relates to the input X. This architecture brought me to 0.01305 rather easily but then I was stuck there for a week.

The NN architecture was initially 2 layers of 128 nodes with output node of 300 which then changed to 3 layers of 64-64-128 with the output of 300 nodes. The 300 nodes will learn different quantiles. For example, a 10 nodes output will learn quantiles of (np.arange(10) + 0.5) / 10, so 300 will learn a much more granular quantiles. I tried 400 but the gain is small.

The choice of loss was using quantile huber loss but later I changed to a simpler quantile loss. Since the mse part of huber didn't seem to help so I only use mae loss on all the quantiles.


The output looks like below: where X axis is the quantiles and Y is the yards prediction.


Now the problem with the output is instead of probability of 199 yards it instead gives 300 output of yards each of the point corresponds to the quantile. So this quantiles need to be converted to each class probability and made a function to do that.

The result is smooth curves predicting the probability of Yards:



For features, I tried to estimate the players position every 0.1 seconds up to 4 seconds. Speed is increasing with acceleration and top speed is limited based on weight. Also correcting trajectories for example if rusher Y is at boundary then rusher will turn right.

Most useful feature from that is mostly distances to nearest defender, how many defenders within certain yards.

Also evaluated the defenders related to the rusher, count of defenders in within 30 degree in front of rusher is important as well.
