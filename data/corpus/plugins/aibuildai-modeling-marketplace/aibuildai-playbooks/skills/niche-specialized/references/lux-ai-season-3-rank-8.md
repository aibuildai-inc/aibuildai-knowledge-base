# 8th Place Solution - Imitation Learning

Competition: lux-ai-season-3
Rank: #8
Source: https://www.kaggle.com/c/lux-ai-season-3/discussion/570673

### Introduction

First, I'd like to thank the team of Lux AI and Kaggle for organizing this competition. I especially enjoyed the new features such as the dynamic environment and the presence of FoW. I look forward to competing with you all in the (hopefully) upcoming fourth season of Lux AI!!

The source code is available here: https://github.com/gregorlied/lux-s3/tree/main

### Overview

My solution is an IL-based approach using the episodes from the teams Flat Neurons and Frog Parade. 
Similar to the teams aDg4b and YumeNeko, I employed two distinct UNet models:

- **Action Model**: A map-level UNet that predicts the action for all units simultaneously.
- **Sap Model**: A unit-level UNet that predicts the sap target for a single unit with action sap.

The models were trained on a dataset containing 11681 episodes from 10 different agents. 

However, different agents might have different action distributions, introducing instabilities during training. Additionally, the model architecture does not inherently leverage the map symmetries, leading to inefficient data usage. To address these issues, I applied two tricks both during training and inference:

- **Agent ID Conditioning:** The model is conditioned on the agent ID both at the bottleneck of the UNet and the prediction head. During inference the model is conditioned to imitate the best agent from team Frog Parade.
- **Game State Normalization:** If the game is played with the POV of player 1, the game state is mirrored along the main diagonal to ensure that the game is always played with the POV of player 0.

Due to the presence of FoW and hidden reward nodes, maintaining a global game state was essential. This state was iteratively updated with local information observed at each turn. For reward node detection, I used the algorithm proposed in [Relicbound](https://www.kaggle.com/code/egrehbbt/relicbound-bot-for-lux-ai-s3-competition).

### Data Selection

I curated a dataset containing 11681 episodes across 10 different agents, selecting only those where the agent won the game. For these episodes, I only retained matches the agent won.

Since the agents of the teams Boey and  aDg4b both stopped playing once a game reached a decided state, I removed these matches from the dataset. The action distributions in these cases differed significantly from those in undecided games, potentially degrading model performance.

### Map-Level Features

Based on the maintained global game state, I constructed 18 map-level features.

| Feature | Possible Values | Remarks |
| --- | --- | --- |
| my_unit_positions | {0, 1} | Whether or not one of my units is placed on the cell (1 yes, 0 no) |
| my_unit_energy | [0, 1] | Normalized energy level per unit |
| my_unit_sap_range_map | {0, 1} | Whether or not the cell is inside the sap range of one of my units (1 yes, 0 no) |
| opp_unit_positions | {0, 1} | Whether or not one of the opponent units is placed on the cell (1 yes, 0 no) |
| opp_unit_energy | [0, 1] | Normalized energy level per unit |
| is_visible | {0, 1} | Whether or not the cell is visible by one of my units (1 yes, 0 no) |
| energy | -1 or [0, 1] | Normalized energy level of the cell (-1 for unknown cells) |
| curr_tile_type_empty | {-1, 0, 1} | Whether or not the cell is an empty tile (1 yes, 0 no, -1 unknown) |
| curr_tile_type_nebula | {-1, 0, 1} | Whether or not the cell is a nebula tile (1 yes, 0 no, -1 unknown) |
| curr_tile_type_asteroid | {-1, 0, 1} | Whether or not the cell is a asteroid tile (1 yes, 0 no, -1 unknown) |
| next_tile_type_empty | {-1, 0, 1} | Whether or not the cell is an empty tile after the drift update (1 yes, 0 no, -1 unknown drift / cell) |
| next_tile_type_nebula | {-1, 0, 1} | Whether or not the cell is an nebula tile after the drift update (1 yes, 0 no, -1 unknown drift / cell) |
| next_tile_type_asteroid | {-1, 0, 1} | Whether or not the cell is an asteroid tile after the drift update (1 yes, 0 no, -1 unknown drift / cell) |
| dist_from_center_x | [0, 1] | Distance from the center with respect to the x axis |
| dist_from_center_y | [0, 1] | Distance from the center with respect to the y axis |
| relic_nodes | {-1, 0, 1} | Whether or not the cell is a relic node (1 yes, 0 no, -1 unknown) |
| reward_nodes | {-1, 0, 1} | Whether or not the cell is a reward node (1 yes, 0 no, -1 unknown) |
| reward_map | [0, 1] | Normalized reward map |



### Global Features

Based on the maintained global game state, I constructed 32 global features.

| Feature | Possible Values | Remarks |
| --- | --- | --- |
| steps | [0, 1] | Normalized game steps |
| match_steps | [0, 1] | Normalized match steps |
| next_map_update | -1 or [0, 1] | Normalized number of steps until next map update (-1 unknown drift) |
| unit_move_cost | [0, 1] | Normalized move cost |
| unit_sap_cost | [0, 1] | Normalized sap cost |
| unit_sap_range | [0, 1] | Normalized sap range |
| unit_sensor_range | [0, 1] | Normalized sensor range |
| all_relics_found | {0, 1} | Whether or not all relic nodes are found (1 yes, 0 no) |
| all_rewards_found | {0, 1} | Whether or not all reward nodes are found (1 yes, 0 no) |
| gained_reward_last_X | {0, 1} | Whether or not my agent gained reward points the last X turns for X = 1, 3, 5, 10 (1 yes, 0 no) |
| reward_last_X | [0, 1] | Percentage of reward points gained by my agent over the last X turns for X = 1, 3, 5, 10 |
| team_points | [0..] | Normalized team points (divided by 500) |
| opp_team_points | [0..] | Normalized opponent team points (divided by 500) |
| nebula_tile_drift_speed_X | {0, 1} | Whether or not the drift speed is of type X (1 yes, 0 no) |
| agent_id_X | {0, 1} | Whether or not the agent id is X (1 yes, 0 no) |


### Network Architecture

The UNet architecture is adopted from the [6th place solution from Season 1](https://www.kaggle.com/competitions/lux-ai-2021/discussion/293776).

Since different agents might have different action distributions, I used distinct prediction heads for each agent to stabilize training, an idea originally introduced in the [4th place solution from Season 1](https://www.kaggle.com/competitions/lux-ai-2021/discussion/296938).

In the last week of the competition, I also experimented with a single prediction head similar to the architecture of the [Diffusion Transformer](https://arxiv.org/abs/2212.09748). Each DiT block is conditioned on the agent ID.

My final submissions consist of an ensemble of both network architectures.

### Map-Level Action Model

- **Input**: Volume of shape 18x24x24
- **Output**: Volume of shape 6x24x24 (Classes: Center, North, East, South, West, Sap)
- **Loss**: Cross Entropy Loss
- **Masking**: All cells without a unit were masked during training and inference
- **Hyperparameters**: Epochs 8, Learning Rate 1e-3, Batch Size 128

### Unit-Level Sap Model

- **Input**: Volume of shape 18x24x24
- **Output**: Volume of shape 1x24x24
- **Loss**: Weighted Dice Loss
- **Masking**: All cells outside of the sap range were masked during training and inference
- **Hyperparameters**: Epochs 4, Learning Rate 1e-3, Batch Size 128

### Submission 1 – v73a (Final Ranking: 2035.6)
You can find this submission here: https://github.com/gregorlied/lux-s3/releases/tag/v73a
- **Action Models**: UNet + Separate Prediction Heads, UNet + DiT Prediction Head
- **Sap Model**: UNet + Separate Prediction Heads

### Submission 2 – v73b (Final Ranking: 2030.3)
You can find this submission here: https://github.com/gregorlied/lux-s3/releases/tag/v73b
- **Action Models**: UNet + Separate Prediction Heads, UNet + DiT Prediction Head
- **Sap Models**: UNet + Separate Prediction Heads, UNet + Separate Prediction Head
