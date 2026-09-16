# Imitation Learning: 3rd Place Solution

Competition: lux-ai-season-3
Rank: #3
Source: https://www.kaggle.com/c/lux-ai-season-3/discussion/568494

At the beginning of the competition, I initially attempted to build a rule-based agent. However, by mid-January, I had hit my limits. After spending weeks trying to improve the solution without any progress, I decided to give the Imitation Learning (IL) approach a shot. Several days were spent downloading and preparing the data, followed by just a couple of hours of training. To my surprise, the IL-based bot outperformed my best rule-based agent. This led me to abandon the rule-based approach and switched to IL. While no groundbreaking discoveries in RL were made, the experience was both valuable and enjoyable!

# Solution

My solution is based on two UNets:

1. World-wise Unit-UNet: predicts the action for each of my units.
2. Unit-wise SAP-UNet: predicts the target for the SAP action.

I had originally planned to combine these two networks into a single model, but I never got around to implementing it.

## Unit-UNet

The Unit-UNet takes two types of input:
1. Feature Maps (28x24x24): These maps represent various aspects of the environment and the unit state at the current and previous steps. Some of the feature maps include: unit positions and energy (current and previous step), fleet vision, nebulae, asteroids, node energy, relics, reward points, the duration a node was out of vision.
2. Global Features (17): These are broadcast to the bottleneck of the UNet and include: move cost, SAP cost, SAP range, team points from the start of the match, team points from the last step, match step, match number, hidden constants (nebula_tile_drift_speed, nebula_tile_energy_reduction, nebula_tile_vision_reduction, unit_sap_dropoff_factor, unit_energy_void_factor).

The output of the Unit-UNet is a tensor of shape 6x24x24, representing the probabilities of performing each of the 6 possible actions (Center, Up, Right, Down, Left, SAP) at each position.

This architecture can't properly handle situations when two or more units occupy the same position. To address this, during training, if multiple units are at the same position, I randomly select one action for all units, prioritizing moving actions and SAP actions over Center actions. This ensures that the model learns to avoid passive behavior and encourages more strategic actions.

During inference, the Unit-UNet predicts a single action for each position. To handle multiple units at a position, I sort them by energy and assign the predicted action to the top half of the units with the most energy. This helps spread units and reduces the risk of clustering. Since top teams typically avoid bunching their units, this limitation of the Unit-UNet isn't a significant issue in practice.

## SAP-UNet

This network complements the Unit-UNet by predicting the target location for the SAP action, while the Unit-UNet determines whether a unit should perform the SAP action in the first place.

The SAP-UNet has a similar architecture to the Unit-UNet, with a few differences. This network is unit-wise, meaning it focuses on individual units rather than the entire environment. In terms of feature maps, I added the unit position and unit SAP positions to help the model focus on the specific location. Additionally, I included unit energy as a global feature.

The output of the SAP-UNet is a tensor of shape 24x24, representing the probability distribution for potential SAP action targets at each position on the grid.

## Data Selection

For my imitation learning, I used replays from the teams "Frog Parade" and "Flat Neurons". Big thanks to these teams!

I didn't use all timesteps from a replay. If the agent lost a game, I added to the dataset only the matches where the agent won. If the agent won the game, I added all matches from that replay to the dataset. However, I never used matches where the outcome of the game was already decided (i.e., when one team won more than 2 matches). This is because there is a chance that these matches no longer reflect normal gameplay and might not be as useful for training.

## Data Preprocessing

I believe Data Preprocessing is the most challenging and crucial part of IL in this competition, and I spent the majority of my time on this step.

The agent does not have full visibility of the environment — it's operating under a fog of war, meaning it can only see a subset of the game state. Additionally, there are hidden constants and reward locations, that are not directly observable by the agent during the game. Simulating these hidden elements accurately is crucial for training the agent to mimic the behavior of the replay agent effectively. To achieve this, at each replay step, I run my own code that receives the information available to the replay agent. The code attempts to identify reward positions, populate the obstacle map (asteroids and nebulae), and uncover hidden constants based on the agent's observations. This process is almost the same as the space.update method from the [Relicbound bot](https://www.kaggle.com/code/egrehbbt/relicbound-bot-for-lux-ai-s3-competition).

As a result, the actual training data is not directly from the replays, but rather the data that my code extracts from the replay agent's observations during the game.

Additionally, if the replay agent's spawn position was in the bottom-right corner, I mirrored the entire map along the line of symmetry. This transformation ensured that all the data in my dataset had a spawn location at position (0, 0). However, this process affected the distribution of Right, Left, Up, and Down actions, increasing the frequency of Right and Down actions while reducing the frequency of Left and Up actions, making the dataset more unbalanced. To address this, I used weighted cross-entropy loss during training, though I’m unsure whether the weights had a significant impact. I also dropped 95% of all instances where all units performed the Center action to reduce their frequency and speed up the learning process.

## Source Code

You can find my code here: https://github.com/w9PcJLyb/lux3-bot

And both of my final submissions are here: https://github.com/w9PcJLyb/lux3-bot/releases/tag/0.4.8

- lux3_0.4.8_fp.tar.gz (submission_id 43374110): Trained on replays from team "Frog Parade"
- lux3_0.4.8_fn.tar.gz (submission_id 43380878): Trained on "Frog Parade" and fine-tuned on "Flat Neurons"

In my local evaluation and on the public leaderboard, lux3_0.4.8_fn.tar.gz performed better.
