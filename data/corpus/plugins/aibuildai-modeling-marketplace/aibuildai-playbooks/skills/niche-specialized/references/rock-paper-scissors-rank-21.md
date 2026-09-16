# 21st Place Solution - neibyr

Competition: rock-paper-scissors
Rank: #21
Source: https://www.kaggle.com/c/rock-paper-scissors/discussion/221597

First of all, I would like to express my gratitude to the organizers for this competition, it was an exciting competition, and very fun.
Also, I would like to mention the participants of the competition who shared their ideas with the community)

**My solution was based on the use of agents, each of which generates a lot of basic agents with different strategy games:**
1. When generating a move, play as an opponent or play against an opponent, in other words, count your moves or your opponents ' moves as your previous moves.
2. Each of the agents generated the probabilities of the move, while using softmax or sum normalization as the normalization
3. Use the action probabilities for the generation or distribution
4. Lag range-generate moves with a delay
5. Shift - after generating the expected move, make a shift

As a result, each of the agents generated a set of sub-agents that played according to their own strategy.

**How did each of the techniques help?**
1. If we get on the opponent, and well understand how he moves, playing as well as he does, then we will predict his moves
2. When generating a move, we can generate the probabilities of moves in different ways, when using softmax, we are more likely to choose moves that have a higher value in the distribution. Conversely, if we are more easily predicted by our opponent in this way, we can switch to an agent that outputs probabilities via sum normalization.
3. All agents with rps_contest predict action.  Also, for the rest of the bots, we can choose the most likely event, and re-arrange the probabilities using this method. For such agents, the probability is generated as follows ` p = np.random.random(3); p[action] += 3 `
4. Playing with a delay helps predict players who may have a bug in the code and also play with a delay, for example, updating the current history after generating a move, without using the opponent's last move.
5. Shift helps you make moves that will beat your opponent, if you have learned to predict his moves, or if your opponent has learned to predict you, you can use this, make a shift of your predictions

**Using these techniques, the following were selected as agents:**
1. Markov agent-which works with different key lengths, and generates a distribution using a weighted sum of predictions from each key. At the same time, three types of agents are used as attributes - by own moves, by opponent's moves, and by moves of both players.
2. rps_contest agents, which are 22 bots
3. SklearnAgent - there features based on public kernel, and we can use models with sklearn-like api. 
4. Iocaine - public kernel

**And using all these agents, the meta agent - MultiArmedAgent-plays**
This agent gives a score to each of the subagents, based on how they play against the opponent.
This score is calculated for the last WINDOW steps and is equal to:
`reward + win_streak * 5 + tie_streak * 1 + lose_streak * (-10)`
Thus, we look at how fast the agent is now, add a reward for the current win_streak, or tie_streak, or fine it if it has a lose_streak
Streak is considered as the number of consecutive game results divided by 5

Using this evaluation method, the TOP-N sub-agents are selected from all sub-agents, on the basis of which the action will be generated

 **Additional techniques**
For additional techniques, I would refer to random
Final agent, played randomly under certain conditions:
1. If we are at the start of the game, in order to collect statistics about the opponent and not give out our strategy, we play the first 100 steps randomly
2. If we win with a large margin, we also start playing randomly, so that we can not use this bot for post analysis
3. If we start to lose, then our strategy was recognized and we have to bring down the current statistics of the opponent, playing randomly
4. With a probability of 0.25, we randomly play any move, regardless of whether the previous conditions are met

Also, I would like to mention that it was not possible to use "Staking" when MultiArmedAgent is also an agent. In this case, the agent played more randomly and could only defeat the base agents, and played in a draw with stronger opponents.


That's all, the code can be viewed on GitHub:
https://github.com/KirillTushin/rps_kaggle
Once again, I would like to thank everyone for this exciting competition!)
