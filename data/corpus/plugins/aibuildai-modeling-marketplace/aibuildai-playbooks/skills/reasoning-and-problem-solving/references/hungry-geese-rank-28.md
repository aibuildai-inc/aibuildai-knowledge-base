# 28th place solution

Competition: hungry-geese
Rank: #28
Source: https://www.kaggle.com/c/hungry-geese/discussion/263709

Thank you for nice competiton!
I appleciate hosts and participants.

I won my first silver medal and experienced reinforcement learning for the first time!

My solution is based on HandyRL and Monte Carlo Tree Search.
Thank you for nice notebook https://www.kaggle.com/yuricat/smart-geese-trained-by-reinforcement-learning and https://www.kaggle.com/shoheiazuma/alphageese-baseline, and congratulations on both authors.

# Solution

- 5000epoch and large minibatch of handyrl's self training ( best score: 1112)
- ensemble some epoch trained models (score: around 1100)

I am newer to reinforcement learning, so I tried various methods around in the dark.
From this competiton, I learned that RL needs more epochs and post processing like MCTS.

# Not work

- increasing gamma and changing other parameters
- expanding model
- centering input by player position
- ensemle without MCTS

# Regret

- I didn't try imitation learning
- Increasing training epoch

Some solutions using imitation learning are published, so I want to try next RL competition!

Thanks you!
