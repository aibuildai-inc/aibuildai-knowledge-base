# 3rd Place Solution

Competition: nfl-big-data-bowl-2026-prediction
Rank: #3
Source: https://www.kaggle.com/c/nfl-big-data-bowl-2026-prediction/writeups/3rd-place-solution

First of all, thanks to NFL, Kaggle, and all organizers for the data and compute—this is the only place we could test our model on the field.

And thanks to my teammates for grinding features, tuning hyper-params, and fixing bugs day and night; the 3rd-place trophy belongs to all of us.

Below, I’ll walk through our solution in these parts: 

1. Feature Engineering
2. Dataset Construction
3. Data Augmentation
4. Use of Additional Data
5. Pre-training & Fine-tuning
6. Model Architecture
7. Losses
8. Training Regularization & Optimization
9. Test-Time Augmentation (TTA)
10. Model Ensemble
11. What Did NOT Help

# Feature Engineering

### Raw Features

x, y, s, a, dir, o, ball_land_x, ball_land_y, num_frames_output

### Velocity Decomposition

velocity_x, velocity_y

### Angle / Motion Direction

angle_to_ball, dir_rad

### Player Role / Mask

player_side_bool, is_passer, is_receiver

### Geometric Distance

distance_to_passer, distance_to_receiver, distance_to_ball_land, passer_to_ball_land, receiver_to_ball_land

### Geometric Projection / Area

distance_to_passing_line, projection_on_passing_line, triangle_area_ratio (area of triangle Player-QB-Receiver)

### Time / Global Context

time_elapsed

# Dataset Construction

We flatten each play along the time axis and lay out the 22 players as fixed slots (missing players are zero-padded). The slot assignment is done once per play:

- At the first frame (frame_id = 1) of every play_id
> a. Offense: sort the 11 players by Euclidean distance to the passer (QB) ascending → slots 0 … 10
>
> b. Defense: sort the 11 players by the same distance ascending → slots 11 … 21

- This ordering is frozen for the entire play: all later frames, features and labels are filled into the tensor using the “nfl_id → slot” map created in step 1.
- The dataset returns tensors of shape (batch_size, time_steps, players, features).
- Late in the competition we found that the exact player order hardly matters; randomly shuffling the players and randomly masking some of them as augmentation gave a solid boost (see “Data Augmentation” section).

# Data Augmentation

- Horizontal flip: mirror left/right across the midfield line (y); offense/defense labels stay the same, routes become symmetric.
- 180° rotation (H + V): flip both x and y; offense/defense labels stay the same, routes stay symmetric.
- Random player dropout
> a. Count valid players from the first frame; skip if ≤ 4.
>
> b. Randomly pick 1–min(5, valid−4) players and zero-out their 36-frame tensor + set mask = False.
>
> c. Push the remaining players to the first k slots and pad the rest with zeros.

- Random player reordering: apply randperm inside offense and defense zones to break the bias that “slot 0 = closest to QB”, forcing the model to rely on geometric features instead of slot id.
- Random input crop: if seq_len > 12, drop 2–6 frames from the head, keep at least 7 tail frames, then pad back to 36 with zeros.

All augmentations are disabled during validation; only raw data is used to keep metrics comparable.

# Use of Additional Data

We incorporated the officially permitted 2018 tracking dataset (https://www.kaggle.com/competitions/nfl-big-data-bowl-2021/data, hereafter “2018nfldata”) into training.

### Data Filtering

The 2018nfldata contains an event column. Under NFL rules we treat the following three events as equivalent to the key moments in this competition:

- ball_snap – start of play
- pass_forward – ball leaves passer's hand
- pass_arrived – ball becomes reachable (ends flight)

We therefore keep only plays whose event chain is
`ball_snap → pass_forward → pass_arrived`.

### Input / Output Split

- Input frames: from ball_snap up to and including pass_forward
- Output frames: from pass_forward + 1 up to and including pass_arrived

### ball_land_x / ball_land_y

Rows with `team == 'football'` give the ball's coordinates.

We take

- `ball_land_x = x` at the row where `event == 'pass_arrived' & team == 'football'`
- `ball_land_y = y` at the same row

### Player Roles

- Passer: player whose `position == 'QB'` in 2018nfldata/players.csv
- Targeted Receiver: offensive player closest to (ball_land_x, ball_land_y) at `event == 'pass_arrived'`

# Pre-training & Fine-tuning

We use a two-stage pipeline:

- Pre-train on 2018nfldata + 2026 competition data with a small, trusted feature set:
`x, y, s, a, dir, o, ball_land_x, ball_land_y, distance_to_passer, num_frames_output, player_side_bool, velocity_x, velocity_y, time_elapsed` (The offline validation score of the Pretrain model ranges from 0.62 to 0.68)
- Build a new model that adds a small linear “bridge” layer (see Model Architecture section), load the pre-trained weights, then fine-tune with the full feature stack described in the Features section, again on 2018nfldata + 2026 competition data.

# Model Architecture

### Key design points

1. Dual-path: Graph-Transformer path captures inter-player interactions ([ref](https://www.kaggle.com/competitions/nfl-big-data-bowl-2020/writeups/deep-snappers-public-5th-place-solution-overview)); main path captures individual motion patterns. 
2. Positional encoding: separate time + player embeddings to keep spatio-temporal order.
3. Masking: handles variable number of valid players per play.
4. Multi-task: four output heads supply rich supervision.
5. Residual links: every projection layer has skip connections to avoid vanishing gradients.



### Other Notes

1. Dropout: set to 0 in every module; for pure regression, any dropout hurts.
2. Width vs depth: wide & shallow (hidden_size=384, num_layers=2) beats narrow & deep (hidden_size=128, num_layers=6).

# Losses

### Primary Loss

TemporalHuberAdapted 
- Time-decay weight: e^(−0.03·t)
- Velocity smoothness (1st-order diff)
- Acceleration smoothness (2nd-order diff)

### Auxiliary Losses

- Inter-frame displacement loss

Self-supervised next-frame prediction inside the input sequence.

1. Predict changes from every input frame to later input frames
2. Goal: temporal consistency
3. Supervision: real frame-to-frame deltas already in the input

> a. i = 0: predict whole sequence → global motion trend
>
> b. i = T-1: predict from last frame → instantaneous change
>
> c. mid frames: learn multi-scale motion

- End-point prediction loss

Force the model to “know” the final target at every time step.

1. Predict displacement from current input frame (i = n) to final target (o = T)
2. End-point awareness at each step
3. Path-planning prior
4. Global convergence guarantee

- Full spatio-temporal correspondence loss

1. Predict displacement from any input frame (i = n) to any output frame (o = m) – all pairs
2. Dense supervision over every possible (n, m) pair
3. Consistency constraint: predictions from different starting points must agree

| Aspect        | Inter-frame         | End-point                 | Full spatio-temporal        | 
| ------------- | ------------------- | ------------------------- | --------------------------- |
| Target        | input → later input | input → final target      | input → any output          |
| Temporal link | causal chain        | single direction (to end) | fully connected             |
| Main role     | motion modeling     | end-point guidance        | spatio-temporal consistency |
| Compute cost  | medium              | low                       | high                        |
| Improved (LB)  | 0.02                    | 0.01                      | 0.003                     |

### Total loss

loss = 1 × primary + 1 × inter-frame + 0.1 × end-point + 0.1 × full-correspondence

# Training Regularization & Optimization

1. AdamW
2. Layer-wise LR: linear layers 3e-4, Transformer 5e-5
3. Cosine annealing down to 1e-5
4. Gradient clipping max_norm = 1.0
5. EMA (decay = 0.999) for validation & final model – fine-tune stage only

# Test-Time Augmentation (TTA)

At inference we apply two augmentations:

1. Horizontal flip
2. Random input crop

Single-model prediction is a weighted average:

`pred = 0.50 * pred_orig + 0.25 * pred_tta1 + 0.25 * pred_tta2`

We did not achieve improvement by adding more augmentations.

# Model Ensemble

We train seven models with game_id 7 GroupKFold offline: (50 epochs / model), cv is about 0.469. And we got LB score 0.458.

In the end, we achieved a slight improvement by ensembling four 7-fold cross-validation models (difference seeds). Public LB score after simple averaging: 0.456.

# What Did NOT Help

1. More Geometric Distance / Geometric Projection / Area features
2. Re-ordering players by distance to receiver instead of QB
3. Extra augmentations: small-angle rotation, per-step noise, temporal mask + linear fill
4. Larger model
