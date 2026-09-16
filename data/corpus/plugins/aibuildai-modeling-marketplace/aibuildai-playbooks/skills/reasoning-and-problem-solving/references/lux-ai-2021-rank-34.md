# 34th Place Solution

Competition: lux-ai-2021
Rank: #34
Source: https://www.kaggle.com/c/lux-ai-2021/discussion/294003

First, Thank you for organizing such an interesting competition.
I spent more than half of the competition period on Reinforcement Learning(RL), but it didn't go as well as I had hoped.
In this discussion, I'll share my best score solution and my RL efforts.

## Solution
My best score agent is created by Imitation Learning(IL) of Unet.
My solution relies mostly on information from nosound(@zaharch). Once again, thank you nosound for sharing useful information with us.
[LuxAI-Solution]

### Collection of episodes
Collect the top team episodes using the notebook shared by Robga(@robga).
https://www.kaggle.com/robga/simulations-episode-scraper-match-downloader

- target team: Toad Bridage
- LB score over 1900
- only win game
- about 1000 episodes(3sub)

### Data Sampling
In order to learn nothing actions, if an actionable unit is not doing anything in episode logs, it is considered a center action and the data set is created like this code.
```python
def extract_center_actions(obs, actioned_unit_ids):
    center_actions = []
    for update in obs['updates']:
        strs = update.split(' ')
        input_identifier = strs[0]
        if input_identifier == 'u':
            unit_id = strs[3]            
            team = int(strs[2])
            cooldown = float(strs[6])
            if (team==obs['player'])&(cooldown==0)&(unit_id not in actioned_unit_ids):
                center_actions.append(f'm {unit_id} c')
    return center_actions
```

Then, in order to reduce the bias in the number of data per turn, only 4 actions per turn were extracted as training data.
At this point since the number of center actions is very large, I under-sampling the center actions  by the average number of other actions.

### Model
My model is created from the following information shared by nosound.
https://www.kaggle.com/c/lux-ai-2021/discussion/289540
https://www.kaggle.com/c/lux-ai-2021/discussion/290284

Since this model was originally created as a pre-training model for RL, it also implemented a value network using Unet's bottle features.

### TTA
The TTA was implemented with reference to [Hangree Geese 4th Solution](https://www.kaggle.com/c/hungry-geese/discussion/263690). Although I did not see a significant change in the score, I adopted it to increase robustness to changes in the map.


### Faster inference by Onnxruntime
I used onnxruntime instead of torch model or torch.jit model to speed up inference and to allow time for the RL matchup and MCTS search described below.

---

## RL approach(not work)
I spent a lot of time on RL approaches, but they didn't work very well. 
I used [PPO on stable baselines3](https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html?highlight=PPO).
Main approach is here.

- Searching good achitecture by IL
RL takes a long time to learn, and trial and error is not easy. So I searched for a good model architecture and observation features by IL.

- Hyper parameter tuning
When the pre-training model by IL was passed to RL, the performance tended to be weak, as if the original strategy was forgotten soon after the start of learning. 
So I tuned PPO's `clip_range` and `target_kl` so that the strategy would not change significantly. This clearly reduced the deterioration of the strategy at the start of learning.
  
brown: default hyper parameter
red: clip_range=0.1 / targe_kl=0.003
[スクリーンショット 2021-11-22 12 22 02]

- Limit the parameters to be learned
I thought it would be inefficient to learn all the parameters of the model, so I decided to learn only the final layer.

- Simple reward
If the number of city tiles or units were used as rewards, the learning would not be stable due to large fluctuations, so I set only the WIN or LOSE of the game as rewards.

- Opponent agents
Self-play and IL agents(LB score is 1400~1500) were employed as opponents.
The ratio are following.
0.25: self-play (current model)
0.15: self-play (random sampling from saved old model)
0.2: IL model by Toad Brigade v1 (LB score is 1550)
0.2: IL model by Toad Brigade v2 (LB score is 1450)
0.2: IL model by RL is all you need(LB score is 1450)

In addition to RL, I tried MCTS by used value of RL model, but it did not work due to the difficulty of implementation. So I'm looking forward to the solution of the team that RL and MCTS worked on.
Thank you for reading!
