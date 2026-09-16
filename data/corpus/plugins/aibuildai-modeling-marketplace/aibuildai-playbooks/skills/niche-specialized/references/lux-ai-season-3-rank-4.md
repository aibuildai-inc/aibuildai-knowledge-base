# 4th Place Solution - Imitation Learning Approach

Competition: lux-ai-season-3
Rank: #4
Source: https://www.kaggle.com/c/lux-ai-season-3/discussion/569928

First of all, thank you to everyone involved in this competition.  
This was my first simulation competition, and it was incredibly exciting to watch my agent grow stronger over time.  
The source code is available here : https://github.com/KASSII/Kaggle_LuxAI-s3

# Overview


- My solution is based on Imitation Learning (IL) and combines the outputs of the following two IL models to determine the behavior of each unit.
  - **Action Model**：A model that predicts which of Move (5 directions including center)/Sap movements will be performed for each unit.
  - **Sap Target Model**: A model that predicts sap targets for each unit that chose the sap action.

- For each IL model input, I used information that can be directly observed from the environment (such as unit positions and energy, etc...) as well as information obtained by updating the map data through exploration (such as point node, etc...).

# Feature extract
- Extract 24x24 map features and scalar-valued global features from the observations obtained at every step.
- For map features with symmetry, I leveraged this property to obtain information about unobserved coordinates (features marked as "Consider symmetry").
- Some features cannot be directly observed, so they were estimated through exploration (features marked as "Estimated feature").
  - The estimation logic is simple, but difficult to describe concisely in prose. The source code will be made public soon; please refer to it for implementation details.

**Map Features**
| Feature name | Estimated feature | Consider symmetry | Memo |
| --- | --- | --- | --- |
| self_unit_pos |  |  |  |
| opp_unit_pos |  |  |  |
| self_energy |  |  |  |
| opp_energy |  |  |  |
| self_enable_move |  |  | Whether or not each of self units has the energy to perform a Move action (1 if possible, 0 if not) |
| opp_enable_move |  |  | Whether or not each of opp units has the energy to perform a Move action (1 if possible, 0 if not) |
| self_enable_sap |  |  | Whether or not each of self units has the energy to perform a Sap action (1 if possible, 0 if not) |
| opp_enable_sap |  |  | Whether or not each of my units has the energy to perform a Sap action (1 if possible, 0 if not) |
| tile_type | ✓ | ✓ | By estimating nebula_tile_drift_speed, I determined the steps when shifts occur and incorporated all tile information that had been explored. |
| visible_mask |  |  | Tile where the unit is visible in the current step (1 if visible, 0 if invisible) |
| map_energy |  | ✓ |  |
| relic_nodes |  | ✓ |  |
| point_prob_map | ✓ | ✓ | A probability map representing the estimated point node locations, inferred via exploration. |
| pre_self_unit_pos |  |  | Position of the self unit observed one step earlier |
| pre_opp_unit_pos |  |  | Position of the opp unit observed one step earlier |


**Global Features**
| Feature name | Estimated feature | Memo |
| --- | --- | --- |
| self_reward |  | Self points obtained in the current step |
| opp_reward |  | Opp points obtained in the current step |
| match_steps |  |  |
| match_round |  |  |
| self_team_point |  | Cumulative self points earned in the current round |
| opp_team_point |  | Cumulative opp points earned in the current round |
| self_team_win |  | Number of matches self won |
| opp_team_win |  | Number of matches opp won |
| unit_move_cost |  |  |
| unit_sap_cost |  |  |
| unit_sap_range |  |  |
| unit_sap_dropoff_factor | ✓ | Estimated from the energy reduction of enemy units adjacent to sap. This feature is only used after estimation (see the Model Switch section for details). |


### Model Switch
The unit_sap_dropoff_factor was a crucial factor for performance improvement. However, since it is a hidden parameter, its value remains unknown until sap is actually executed and the energy reduction of enemy units is observed, making it impossible to use beforehand.
To address this, I trained two versions of both the action and sap target models:
 - One without unit_sap_dropoff_factor in the Global feature
 - One with unit_sap_dropoff_factor included

Initially, the model without unit_sap_dropoff_factor was used. Once the value was estimated, I switched to the model incorporating it. This model-switching approach enabled adaptive selection of the optimal model at each stage of the game.

# Imitation Learning
## Data
- I used only the 'Frog Parade' team's replays for Imitation Learning, focusing on winning episodes based on early experiments.
  - Due to a temporary bug in the matching system, I filtered episodes by selecting opponents with leaderboard scores above a certain threshold.
- For the action model, I used data from all steps of the filtered episodes, while for the sap target model, I extracted and used only the steps where units selected a sap action.


## Network Architecture


- I adopted a simple UNet-based architecture, referencing the [6th solution from Season 1](https://www.kaggle.com/competitions/lux-ai-2021/discussion/293776).
- The map features are used as inputs to the U-Net, while the global features are broadcasted and concatenated into the bottleneck part of the U-Net.

## Details of each model
### 1. Action Model
#### input-output
- The input is a 15-dim map feature and a 12-dim global feature as described in the Feature Extract section above.
- The output is a map of (6, 24, 24) representing the probability of the action (center, up, right, down, left, sap) that the unit at each coordinate should take.

#### training
- loss
  - Since this game allows unit overlap, it can be considered a 6-class multi-label classification task, so I used BCEWithLogitsLoss.
  - During loss calculation, I applied a masking process to ensure that only the loss corresponding to the coordinates where units exist is valid.

- augmentation
  - Rotations  (0°, 90°, 180°, 270°) were applied with equal probability.
  - Vertical flip

- fine-tuning
  - First, I trained the base model using multiple episode datasets from Frog Parade's various submission IDs (collected from mid-February to just before the final weekend, totaling 19787 episodes).
  - Next, I fine-tuned the base model's weights using episodes from Frog Parade's best LB submission ID as of the final date (1046 episodes).


#### inference
- TTA
  - I applied TTA with 4 rotations (0°, 90°, 180°, 270°) and vertical flip (on/off), averaging the 8 outputs.

- thresholding low-probability actions
  - If actions are selected solely based on the probabilities output by the model, there is a risk that low-probability actions may still be chosen.
  - Therefore, I modified the selection process to consider only actions whose cumulative probability exceeds a threshold (experimentally set to 0.7 in this case).

- adjustment of probability when multiple units exist at the same coordinates
  - In this model, which is trained as a multi-label task, when multiple units overlap at the same coordinate, the probability map outputs equal probabilities for each unit's actions. As a result, even in situations where the units should ideally take different actions, they may probabilistically end up selecting the same action.
  - I addressed this issue by adopting the following selection method:
      1. The first unit selects an action based on the original probability distribution as usual.
      2. The probability of the selected action is reduced by 1/(number of units at the same coordinate), and the remaining probabilities are normalized.
      3. The next unit selects an action based on the adjusted probability distribution, reducing the likelihood of selecting the same action as the previous unit.



### 2. Sap Target Model
#### input-output
- The global features are identical to those used in the Action Model (12 dim).
- For map features, I excluded past step information (pre_self_unit_pos, pre_opp_unit_pos) and instead added target_unit_sap_area and other_unit_sap_area.
  - The past step information was excluded because local experiments showed that removing it resulted in a higher win rate.


- The output is a (1, 24, 24) probability distribution where each coordinate represents the potential dominance of the sap target of the target unit.

#### training
- loss
  - Since this can be considered an N-class classification task (where N = 24*24 = 576), I used cross-entropy loss.

- augmentation
  - The same as the Action Model.

- fine-tuning
  - The same as the Action Model.

#### inference
- TTA
  - The same as the Action Model.
- thresholding low-probability actions
  - The same as the Action Model.
  - After some experimentation, the threshold for the sap target model was set at 0.6.


# What didn't work
- other data selection methods
  - I experimented with using data from winning matches even when losing the game, mixing data from another team, and other variations, but did not show clear improvement.

- Use of long-term time series information
  - I attempted to incorporate LSTM and Conv3D to leverage long-term temporal information, but the performance deteriorated significantly.
  - Although leveraging long-term past information failed, stacking unit position information from the previous frame improved performance in the action model, so I adopted this approach.

- Conversion to Reinforcement Learning
  - Since IL was shown to work reasonably well, I attempted to use the IL model as an pretrained weight for RL. However, the RL approach did not succeed.
  - I experimented with various parameters, however, the results were either a complete collapse of the IL policy or no improvement over the default behavior.
  - After 2–3 weeks of trial and error without success, I concluded that with my resources and implementation, it was nearly impossible to train for a sufficient number of steps. As a result, I ultimately focused solely on IL for development.
