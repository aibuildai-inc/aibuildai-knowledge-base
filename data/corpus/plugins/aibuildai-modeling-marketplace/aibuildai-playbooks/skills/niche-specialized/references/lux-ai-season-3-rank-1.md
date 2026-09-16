# 1st place approach by Flat Neurons

Competition: lux-ai-season-3
Rank: #1
Source: https://www.kaggle.com/c/lux-ai-season-3/discussion/569562

Hello and welcome! Here’s an extensive overview of our design, training pipeline, and testing strategy for the Lux AI Season 3 challenge.

## Core Idea

We used multi-agent reinforcement learning (RL), primarily employing the IMPALA algorithm with enhancements like dynamic reward scaling and adaptive entropy.

## Action Space

We ended up with the following action space for **each** of the 16 units:

### First Head (Movement + Sap decision):
- **0**: Do nothing (stay in place).
- **1**: Move left.
- **2**: Move right.
- **3**: Move up.
- **4**: Move down.
- **5**: Sap (ranged “attack”).

### Second Head (Sap target selection):
- If the first head predicts “sap,” we then look at the second head to decide **which tile** to sap.
- This second head predicts a target in a 15×15 square around the unit (a ±7 range, roughly) so that the unit can choose any tile within that bounding box as the sap target **without masking**.
- The second head is trained **only** on timesteps where the first head selected “sap,” while the first head is trained on **all** timesteps.

**In inference**:

- We **query all heads every timestep**, but we only **use** the second head’s output if the first head’s action is “sap.”
- If the chosen action is “sap,” we look at the second head for the exact tile to sap.
- If it’s anything else (move or stay still), we ignore the second head’s output for that unit.

---

## Observation Space and Features

We used roughly **1000+ features** per tile on the 24×24 map. About 100 of these were continuous, while the rest were one-hot encoded versions of discrete features (e.g., presence of an asteroid, which team’s unit is on a tile, how many steps ago we saw an enemy at that tile, if that tile is part of a relic node region, etc.).

- **Temporal Info**: Some features tracked how many steps had passed since we last observed a tile or last saw an enemy on that tile, etc.
- **State Tracking**: We ran an internal state updater that aggregated discovered parameters over time. For instance, if we started suspecting that `nebula_tile_vision_reduction = 3`, or we found what tile gives points, we stored that internally and used it to calculate features.
- **Continuous + Discrete Encoding**: Each feature was encoded both in a normalized continuous form **and** by discretizing it (through binning) into a one-hot vector.

We also had a specialized head for **predicting the positions of enemy units on the next tick** — even for tiles we currently cannot see. This was a supervised head using a weighted binary cross-entropy loss, trained from partial ground truth. We’ll explain this in more detail below.

---

## Network Architecture

[network architecture]

1. **Input Encoding**  
   - We start with the ~1000 binary features + ~100 continuous features for each of the 24×24 cells, plus 3 additional “enemy future prediction” features from the previous step.
   - That entire set (for each cell) is compressed down to an internal representation of size ~500, and then further compressed to 128 channels per cell.
   - Now we have a 24×24×128 feature map.

2. **Residual and ConvLSTM Layers**  
   - We apply **24 residual blocks** on the 24×24×128 volume (no shape changes).
   - Next, we apply a **ConvLSTM** (with hidden state size 24×24×128) that keeps track of temporal information across timesteps. This means our model effectively has memory from previous steps.

3. **Transformer**  
   - We then apply **4 blocks of a Transformer** across the entire map (24×24 positions). Essentially, each cell can attend to other cells (spatial attention) to capture long-range interactions.
   - Let’s call the result of this pipeline **BASE_OUTPUT**, which is again 24×24×128.

4. **Heads**  

   - **Enemy Future Prediction Head**  
     - We feed BASE_OUTPUT into a head (which is a simple conv) that outputs a 24×24×3 tensor, representing predicted:  
       1. The probability that an enemy unit is in a tile next tick.  
       2. The probability that a tile will be adjacent to an enemy unit’s position next tick.  
       3. The combined sensor mask of all enemy units next tick.  
     - This was trained in a supervised manner.
     - Output of this head then passed to action head.

   - **Value Function (Baseline) Head**  
     - We feed BASE_OUTPUT into a small Transformer that can incorporate knowledge of both teams  and produce a scalar value (the expected future return). This is used as a baseline for policy gradient methods (IMPALA / V-trace).

   - **Action Head (Movement + Sap Heads)**  
     - **Patch Extraction**: For each of the 16 units, we extract a 15×15 patch from BASE_OUTPUT combined with prediction head output centered on the unit’s position. Positions outside the map were marked with an additional feature set to 1.  
       - That yields a shape of **16 × 15 × 15 × ~128** (one patch per unit).  
     - We also have a set of per-unit features (energy, the unit id in tile, etc.), which appended to each per-unit patch pixel, and a then get **processed using a small MLP** . This results in a final shape like **16 × 15 × 15 × ~150**.

     - **Sap-Target Head**:  
       - A straightforward convolution-based approach on top of the 16×15×15×~150 patches to produce a single logit for each of the 15×15 possible sap-target cells.  
       - So, for each of the 16 units, we get a **15×15** logit map for sap actions.

     - **Movement Head**:  
       - We apply a few residual blocks (4 blocks) with channel compression to produce a final 6-dimensional output for each unit’s patch: the 6 possible actions (stay still, move up/down/left/right, or sap).  
       - So effectively we get **16 × 6** logits.

---

## Training Procedure

### Core Algorithm: IMPALA + V-trace

We trained our agent with a large-scale **IMPALA** setup, using V-trace and additional modifications like Upgo, TD losses, and an entropy term. We generated experiences by having the agent compete in an environment that **in some fraction of games** faces its latest self, but also competes against frozen older models or a teacher model (explained below).

When pure self-play plateaued, we introduced extra techniques:

- **Teacher KL and Teacher Baseline Loss**  
  - We kept around a “frozen teacher agent” that was our best model so far.  
  - We added a KL divergence loss between the student’s policy and the teacher’s policy, plus teacher baseline loss align the value heads.  
  - This prevented forgetting of previously learned skills.

- **Frozen Opponent Pool**  
  - Instead of always self-playing the latest version of the agent vs. itself, we also let the agent face a pool of older (weaker or moderate) opponents at some fraction of games.  
  - This improved the agent’s robustness, preventing overfitting to a single self-play style.

- **Behavior Cloning (BC) from external replays**  
  - We implemented posibility to mix RL batches with BC batches from replay data while training, since Frog Parade was dominating, but in the end, we did not incorporate this, since we were already seeing significant improvements without it.

### Dynamic Rewards

During training, we used a reward scaling mechanism to keep the target value range consistent. Typically, our baseline head (the value function) might predict returns in a range like [−5, +5]. But the actual rewards from environment events (e.g., collecting relic points) might be too small or too large at different stages of training.

1. We tracked a sliding average of real returns over ~5000 batches.  
2. We computed a multiplier so that the scaled returns would land neatly inside the [−5, +5] range.  

For example, early on, if your agent rarely scores points, the few times it does score might result in big scaled rewards. Over time, as it regularly scores, the multiplier shrinks to keep values in range.

### Dynamic Entropy

We also used a dynamic entropy strategy for the two action heads (movement vs. sap-target). Instead of setting fixed entropy coefficients, we:

1. Define a **target entropy** for each head that starts high (e.g., 0.9 for movement and 3.9 for sap-target) and decays linearly to 0 across 100M steps.  
2. Force the policy’s actual entropy to track this target by adjusting the entropy coefficient automatically.  

In practice, this means early in training, the agent explores a lot (high entropy). Over time, as the target entropy goes down, the agent becomes more exploitative.

When we continue training beyond 100M steps, we reset the target entropies to smaller initial values (e.g., 0.45 and 2.0) and decay them again. This can cause short-term performance dips but usually yields a stronger final policy.

The graphs below show the training progress of final model last iteration:






---

## Final Rewards

We experimented with many shaped rewards, such as:

- **+** for collecting relic points  
- **-** for opponent collecting relic points  
- **+** for winning a match  
- **-** for losing a match  
- **+** for discovering or removing cells from potential relic-node areas  
- **+** for dealing damage  
- **-** for receiving damage  
- **+-** for death  
- **+** for having a larger sensor range  
- **+** for seeing an opponent  

Eventually, we moved toward a sparse (match result) scheme:

- **±1** for winning or losing the match.  
- **±2.5** summary for full game for points (+ when we collecting, - when opponent collecting)

Concretely:

- If you win all matches and collect the maximum possible relic points (while your opponent collects zero), your final return for that game might be +7.5, and the opponent gets −7.5.  
- Partial scoring for relic points helped prevent “do nothing” stagnation, when agent thinks it will guaranteed win or lose.

---

## Implementation Details: Flips, Data Augmentation, Time Limits, Action Selection

### Always Starting at (0, 0)
We forced our agent to **always start** from \((0, 0)\). If it was actually assigned to the “second player” side, we **flipped** observations and actions so that the agent’s perspective remained consistent. Essentially, any coordinate for the second player was mirrored or rotated to align with the viewpoint of having started at \((0,0)\).

### Kaggle Inference: Flip-Based Data Augmentation
On Kaggle, we went one step further and applied **data augmentation** by flipping x/y coordinates during inference. This can help smooth out any bias from map orientation. However, if we used more than **30 seconds** of overtime, we **disabled** this augmentation to halve inference time and avoid timeouts.

### Action Selection: Training vs. Kaggle
- **During Training**: We sampled from the policy’s probability distribution (stochastic). This encourages exploration.  
- **On Kaggle Inference**: We chose the **most probable** (greedy) action. This maximizes expected performance, avoiding unnecessary randomness in critical matches.

### Comparing Agents: Real-Time Win Rates
We compared agents by running **thousands** of self-play matches. Toward the end, we could also track real-time win rates against both **older versions** and the **teacher** version. This provided rapid feedback, letting us confirm whether each new iteration truly outperformed prior policies.

---

## Hardware and Training Scale

We had substantial computing resources. The final model took ~3–4 days of training for ~1.5B environment steps, done in iterations of 200M steps. Over the entire competition, we totaled over 20B steps across various experiments.

Some performance boosters:

- **bfloat16** for reduced precision training.  
- **PyTorch 2.0 `compile()`** for a ~1.5× speedup.

---

## Public Testing Strategy

One unique challenge was testing new models on the Kaggle leaderboard without allowing certain “Imitation Learning (IL)” opponents to copy our best strategies. 


To begin with, let’s highlight a few observations that motivated our approach to submitting solutions:

1. Looking at other participants’ matches, we saw that certain agents imitate the actions of stronger opponents. Thus, IL might copy your solution.
2. If IL trains well, it could end up playing 50/50 against its donor. But we weren’t certain if IL, trained on a small number of strong players, could actually surpass them.
3. If the model outperforms others significantly, we’d prefer not to reveal that advantage until the end.


At the time, we already had a model that reliably held 3rd place in the rankings and had some execution time to spare.

Based on this, we devised the following plan:

1. Included **two models** in the solution. The first was a Kaggle-tested version already capable of reaching the top. The second was a more powerful model we want to evaluate.
2. When a match starts, we pick the weaker model **85% of the time** to play the entire match. In the **remaining 15% of cases**, we use the stronger model.
3. Noted in the agent log which model participated in each match.
4. Used a Python script to collect the match data, grouping by factors like `main_submission_id`, `enemy_submission_id`, and `is_strong`, then computed the actual win rate against opponents.

Visually,  our dataframe with results looked something like this (where `count` - count of total played games, `sum` - games with win):
[winrate dataframe]

**Key benefits of this strategy:**

1. Because the first model already occupies 3rd place and is used most often, we secure a top rank and still face the opponents we care about.
2. Top-ranked players collected roughly 1,000 matches per day, providing enough data to accurately assess the strong model’s win rate.
3. If IL is trained on our winning games, distinguishing which model (strong or weak) was responsible becomes much harder, especially since they may have different inputs and strategies, complicating IL training further.


Closer to the end of the competition, when we tested larger models, we tweaked our approach slightly because two large models couldn’t fit in a single submission.

Instead, we injected noise into the model’s logits with a fixed probability to weaken our submission. We also began using the stronger version not in separate matches, but across certain rounds within each match.

Here's an example of our stats (enemies winrates against our solutions) based on game episodes before some days until deadline:

[winrate by episodes]

---

## Closing Thoughts

Throughout the competition, our main lessons were:

1. **Large-Scale Self-Play** with advanced model architectures (ResNet, ConvLSTM, Transformers) can yield very strong multi-agent policies.  
2. **Adaptive Reward Scaling** and **Dynamic Entropy** are extremely helpful for stabilizing training signals, especially when the environment is complex and the final objectives (like relic points or match wins) might be sparse or non-stationary.  
3. **Teacher–Student** frameworks prevent losing past knowledge, and **opponent pools** avoid overfitting to the latest version of oneself.  
4. **Careful Testing** is crucial: you want to see real performance on the leaderboard but not necessarily reveal your entire advanced strategy to potential imitators.

In total, we trained for over 20B steps across all experiments, culminating in a final model that leveraged advanced exploration strategies, partial information reasoning (guessing unseen enemies), and sophisticated multi-agent behaviors. We hope this summary gives a thorough look at the intricacies behind building our Season 3 Lux AI bot.
Final solution source and submissions can be found [here](https://github.com/tonykozlovsky/lux-ai3-pub).
If you have any questions or feedback, feel free to reach out! Thank you for reading, and best of luck in your own RL or multi-agent endeavors.
