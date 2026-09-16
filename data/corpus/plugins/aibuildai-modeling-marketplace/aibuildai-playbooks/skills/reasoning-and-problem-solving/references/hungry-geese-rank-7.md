# Team CGILAB x H2O.ai (7th place) solution

Competition: hungry-geese
Rank: #7
Source: https://www.kaggle.com/c/hungry-geese/discussion/263486

Thank Kaggle for hosting this wonderful competition and also the abundant resources in notebooks and discussions. First of all, I would like to thank my teammate @khyeh0719. We get inspired with many interesting ideas in the discussion. I enjoy it and learn a lot during the competition. The following is the solution for our team.


### **Overview: Supervised Learning + Searching + Heuristics**

Since we joined the competition late, we first started with training a supervised learning network from top episodes in the leaderboard. Second, we selected Monte-Carlo tree search (MCTS) for the search algorithm to help the supervised learning network avoid some pitfalls. Third, we designed two heuristics by observing the games we played against other agents in the leaderboard. After combining these three methods, we finally have a chance to enter the gold area.

We picked two agents as opponents to evaluate the performance during development, including "greedy" and "smart_reinforcement_learning" from this notebook: [Hungry Geese - Agents Comparison](https://www.kaggle.com/ihelon/hungry-geese-agents-comparison). The following are the strengths of different methods, 400 games are played for each experiment, the number representing the win rate for our agent reaching the first and the second place at the end of the game.

| Method used | v.s. 3xgreedy | v.s. 3xsmart_rl |
| --- | --- |
| Supervised Leaning | 77.25%, 89.50% | 51.50%, 69.00% |
| Supervised Leaning + Searching | 79.50%, 89.25% | 65.75%, 84.50% |
| Supervised Leaning + Searching + Heurisitcs | 88.75%, 95.25% | 77.50%, 91.00% |

In addition, we planned to use reinforcement learning to improve the playing strength automatically. However, due to the time limit, we didn't finish our RL agent before the end of the competition.


### **Supervised Learning**

We utilized two public notebooks to help us complete the supervised learning network. We are very grateful for these contributors.

**[Training Data]**
We collected 21,000 top episodes from the leaderboard by following this notebook: [
Simulations Episode Scraper Match Downloader](https://www.kaggle.com/robga/simulations-episode-scraper-match-downloader). According to our analysis, over 50% of episodes are played by "Goosebumps" because his army already occupies the top of the leaderboard. As a result, the goal seems to imitate "Goosebumps" when training the supervised learning network.

**[Network architecture]**
We followed the Handy RL network from this notebook: [Smart Geese Trained by Reinforcement Learning](https://www.kaggle.com/yuricat/smart-geese-trained-by-reinforcement-learning), with some modifications as follows.

- We increased the outputs to 4 policy heads and 4 value heads corresponding to each player's view. The reason is that we can save evaluating the network 4 times for each player when using the network in the search. In the training, if an agent is dead, we skip updating its loss.
- We add one more feature plane representing the "current step" because we think the remaining steps are also important information. The "current step" feature is a float number defined by `current_step / 199`, and the plane is filled with the same number.
- We change the value targets to [1, 0, -1, -1] from the first to fourth place respectively. Specifically, we want the network to strive for the first place as much as possible.

For some minor details of training, we used a batch size of 1024, and Adam for the optimizer, the learning rate is fixed at 0.01 in the whole training. We are surprised that the network converged quickly with only a few hundred steps. The network we used is trained only for 150,000 steps with around 82% of accuracy.


### **Searching**

We found the agent often falling into traps if only the supervised network is used. Hence, we tried to perform a search to help the network avoid some pitfalls. Inspired by the AlphaZero algorithm, we designed an AlphaZero-like Monte-Carlo tree search and improved the search speed by some techniques often used in MCTS.

**[AlphaZero-like Monte-Carlo Tree Search]**
We used a decoupled PUCT and skipped new food generation during the search. Following list of details for our MCTS.

- In the selection phase, each agent selects action by maximizing its rewards without knowing the other agent's strategy. We chose 1.25 for the exploration constant in the PUCT formula.
- To reduce branches, we skip the food in the expansion phase. We believe surviving until the end of games is always a better strategy rather than chasing the food.
- In the evaluation phase, we backup the value to each agent individually by the value network. If an agent is dead during the search, we give a terminal reward [1, 0, -1, -1] according to its rank.
- At the end of the search, we pick the move with the highest simulation counts.

In our experiments, we found the agent performs well with around 50 simulations and became stronger until 200 simulations.

**[Optimization for speed]**
We also did some improvements to speed up the search, listing as follows.

- dynamic thinking time: For each move, the thinking time of MCTS will be `1 + remainingOverageTime/remainingSteps`.
- early abort: If the second-highest simulation count of the move is far away from the highest simulation count of the move, then we stop the search immediately in order to save the remainingOverageTime.
- reuse tree: When receiving a new observation, we will reuse the search tree if the other agent's moves have been simulated in our search tree. Note that we will not reuse the tree if there is a new food created since we don't consider the food, which causes some bias in the search.

Overall, with optimization, our agents are able to perform around 50~150 simulation counts per second in Kaggle's contest environment. However, we observed that the speed is very unstable for each game. We don't know if it's because every game is running on different machines.


### **Heuristics**

We designed two heuristics by observing the games we played against other agents in the leaderboard. These heuristics seem tricky but are very useful in our experiments.

**[Terminal Reward]**
In the search, we change the terminal value to [1, -0.2, -1, -1] instead of using [1, 0, -1, -1]. Intuitively, if only two players are remaining, the value from the value network should be greater or equal to 0 for every agent. However, sometimes the value network will give a very small negative value (like -0.001) when there are only two players. This will induce our agent to self-suicide because of avoiding negative values from the network. Hence, we add a little negative value for the second place to differentiate this case.

**[Single-agent against 3-agents]**
Because we use the same network in the search, our agent performs well if all other agents are using a similar strategy. However, we observed that our agents often had a head-to-head collision with others at the beginning of submission. We ease this problem by treating the other agents as a group.

In the search, for the other 3 agents, we add an additional value (0.5) to the prior if the action can move to the neighbor of our agent's head. At the first glance, this will induce the other 3 agents to suicide with our agents. In fact, this will also induce our agents to avoid colliding with other agents at the same time. While the added value only adds on prior, the search tends to be regular when the simulation count grows.


### **Follow up**

Although the two heuristics improve the playing strength a lot, we believe using RL algorithms to learn these heuristics automatically will be a better choice. The next step could be generating stronger episodes by the current agents and improving the network by training the episodes, as the AlphaZero algorithm and this notebook [(AlphaGeese Training)](https://www.kaggle.com/shoheiazuma/alphageese-training) did. Overall, we enjoy this competition and hope everyone can get their best in the end.
