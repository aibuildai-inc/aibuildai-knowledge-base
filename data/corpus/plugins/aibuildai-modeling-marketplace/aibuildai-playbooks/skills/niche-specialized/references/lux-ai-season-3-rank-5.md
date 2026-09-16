# Kiwis xLSTM agent (5th place solution)

Competition: lux-ai-season-3
Rank: #5
Source: https://www.kaggle.com/c/lux-ai-season-3/discussion/571111

## 1\. Introduction

Hi everybody, first I want to thank the organizers and all the helpful people on discord who made this competition really enjoyable\! Additionally I want to thank my University and Institute, who allowed me to use some of their compute resources\! This is the first time I seriously participated in a competition so my approach and code is very messy and my strategy was definitely lacking in many areas. Nevertheless, here's my final solution. Code can be found on [github](https://github.com/Elias-Buerger/kaggle-lux-s3).

## 2\. General

In general I developed an xLSTM-based \[1\] approach, which appeared suitable for this partially observable decision problem. I combined this with RL and a recurrent PPO approach with self play. I used this opportunity to learn JAX and made sure to make the entire training loop jittable. Because we are dealing with a partially observable environment I used a separate actor and (all knowing) critic. The model is based on JointPPO \[2\] which uses a single network to control all units.

## 3\. Visualizer

Something I learnt quickly throughout this competition is to VISUALIZE. Because this is the first time I used JAX and because I coded a lot of the framework from scratch (with big parts inspired by PureJaxRL [3]), my implementation had (and may still have) many bugs. Without adapting the visualizer to give me more information about my models inputs and outputs, I would have never found many of them. Here is an example of what the final changes to the visualizer look like:



I display the global features underneath the map, the map input channels below that and the friendly unit features in the unit list. I can also visualize the "valid sapping locations" mask with a blue tint.

## 4\. Observations and Actions

Here, I want to explain how I encoded the inputs and the models outputs.

**Input encoding:** I differentiate between three types of observations:

* *Global features* (like current match, points, number of units, ...) represented by a single 15-value vector for the actor and 18-value vector for the critic.  
* *Map features* (like vision, relic location probabilities, energy field, unit movement history, ...) represented by a 24x24x11 map for the actor and 24x24x15 map for the critic  
* *Unit features* (like location, last five energy values) represented by 32 5-value vectors \+ 32 positions (one per friendly and opponent unit)

I also add the x and y coordinate to the map features as a positional encoding and flip all features (positions, map, ...) to make it look like the player is always in the top left corner from the model's perspective.

Relic fragment locations are calculated by keeping a history of unit locations and points gains. Relic probabilities are iteratively updated whenever we get new information to exclude / re-include locations. The algorithm isn't perfect but I think the recurrent network is capable of figuring out the rest.

**Output:** The model outputs an action type and a sap location for each unit. Invalid actions like moving outside of the map or sapping an area out of reach are masked out. Units can only sap a 3x3 area around visible opponents or invisible relic fragment locations.

## 5\. Model architecture

I developed an architecture based on xLSTM for time-series modeling and a Transformer for encoding the current state. The concept of the model and training is inspired by JointPPO which views a multi-agent decision problem as a sequence modeling problem, by iteratively predicting actions for each unit. While the actor and critic share a lot of the architecture, they are completely separate models and the critic gets more detailed information (like opponent positions, relic fragment locations, nebula speed, ...). The models look roughly like this\*:



\*There are also a bunch of Layernorms, skip connections and other small details that I have not drawn here.

**Encoder:** The encoder consists of a) ConvNeXt \[4\] to encode the map, b) an encoder for friendly units, c) an encoder for opponent units, and d) an encoder for the global state. 

*Map encoder:* I used a ConvNeXt for image features because I found it to work better than a simple ResNet, while also being faster to train and needing less parameters. Initially the map features pass through 3 ConvNeXt Blocks without reducing the resolution. This way I can take the features at the unit locations and add them to the other unit features. Only then is the map compressed into a single feature vector of length 128 by the rest of the ConvNeXt.

*Friendly and opponent unit encoders:* Friendly and opponent unit features as well as global features each pass through a MLP with gelu activation and Layernorm. All friendly units share one MLP and all opponent units share one MLP. Some parts are shared between friendly and opponent units (e.g. weights for encoding the position, map features and energy values)

*Global state encoder.* Here I used a simple MLP with one hidden layer, gelu activation and Layernorm.

**Transformer meta-encoder:** After encoding all features, I end up with one map token, one (learned) recurrent token, 32 unit tokens and one global token. I feed these through four transformer layers with self attention and a gated MLP, while masking out units that are either dead or haven't spawned yet.

**xLSTM core:** To handle the time-series nature of the game via recurrent neural networks, I use an xLSTM consisting of an mLSTM layer for memory capacity and an sLSTM layer. xLSTM just performs much better than a simple LSTM while also needing less parameters. I re-implemented the xLSTM in pure JAX, which allowed for a full JAX-based pipeline, but the implementation has some limitations concerning efficiency, speed, and memory usage. I pass the recurrent token through the xLSTM before either using it inside the value head or adding it to each friendly unit vector.

**Heads:** The xLSTM model is then complemented with an actor and a critic head to allow for PPO training. The *critic* head is just a MLP with spectral norm. The *actor* head is a transformer decoder that uses the final unit embeddings as queries and the predicted actions as keys and values for cross attention. It also includes a skip connection from the map features to predict sapping locations.

The final actor and critic each have \~2M parameters. But on the kaggle servers only the actor is deployed. For each unit the model predicts a probability distribution for the action type and a probability distribution over each possible sap location. On kaggle, the action and sap location is taken by using the element with the largest probability. Actions are re-encoded and fed back into the transformer.

## 6\. Training

For training I initially trained a much smaller model \~500K parameters with only 1 transformer layer, a single mLSTM layer and by predicting all unit actions at the same time instead of sequentially. The small model was trained in 3 phases:

* On a 16x16 map with shaped reward for 700k steps  
* On the full 24x24 map with shaped reward for 700k steps  
* On the full 24x24 map with sparse reward for 700k steps

After the final stage the model stopped improving. While this was enough to get a gold medal (at least at the time), I wanted to go further so I decided to train a larger model. This one trained in 2 phases:

* On the full 24x24 map with the small model as teacher (by switching the PPO loss to a cross entropy loss between the small teach models logits and the new models logits) for 100M steps  
* On the full 24x24 map with shaped reward for as long as possible (\~800M steps).

I found the model to improve slower when switching to sparse rewards. The model still kept improving even after I submitted the final checkpoint before the deadline. I could have also trained an even larger moder since I never had time management issues on the submission server. However, I didn't know if it was worth it switching to an even larger model in the last week of the competition.

Self play was done by playing against the last 128 checkpoints in 25% of games and against the latest checkpoint in 75% of games. Because JAX allows you to play all these games in parallel, I could keep all the weights on the GPU and just vmap over them.

I used the following hyperparameters for training the final model:

| Parameter | Value |
| :---- | :---- |
| LR | 3e-4 |
| NUM\_ENVS | 1024 |
| NUM\_STEPS\_BETWEEN\_UPDATE | 128 |
| BPTT\_HORIZON | 16 |
| OPPONENT\_UPDATE\_STEPS | 2^20 |
| OPPONENT\_BUFFER\_SIZE | 128 |
| LATEST\_VARIABLES\_ENVS | 768 |
| UPDATE\_EPOCHS | 2 |
| MINIBATCH\_SIZE | 64 |
| GAMMA | 0.997 |
| GAE\_LAMBDA | 0.9 |
| CLIP\_EPS | 0.05 |
| ENT\_COEF | 0.001 |
| VF\_COEF | 0.5 |
| MAX\_GRAD\_NORM | 5 |

All training was done in bfloat16.

## 7\. Conclusion

All in all I had loads of fun, got to learn JAX (and became an absolute fan) and talked to really helpful and cool people on discord.

Concerning the competition itself, I really liked the recurrent aspect of the game and think the balance patch, which introduced relics that spawned later on, was one of the best decisions in this competition, making me want to try recurrent models. I also really appreciate the small map size which made it possible for people like me to start training at home on a small GPU with only 8GB of memory\!

## 8\. References

\[1\] Beck, M., Pöppel, K., Spanring, M., Auer, A., Prudnikova, O., Kopp, M., ... & Hochreiter, S. (2024). xlstm: Extended long short-term memory. *arXiv preprint arXiv:2405.04517*.

\[2\] Liu, C., & Liu, G. (2024). JointPPO: Diving deeper into the effectiveness of PPO in multi-agent reinforcement learning. *arXiv preprint arXiv:2404.11831*.

\[3\] https://github.com/luchris429/purejaxrl

\[4\] Liu, Z., Mao, H., Wu, C. Y., Feichtenhofer, C., Darrell, T., & Xie, S. (2022). A convnet for the 2020s. In *Proceedings of the IEEE/CVF conference on computer vision and pattern recognition* (pp. 11976-11986).
