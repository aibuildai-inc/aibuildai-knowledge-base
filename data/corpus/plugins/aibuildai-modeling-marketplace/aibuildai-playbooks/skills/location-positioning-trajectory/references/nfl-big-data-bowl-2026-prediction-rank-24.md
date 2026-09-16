# Private 24th | Public 23rd solution - Data augmentation

Competition: nfl-big-data-bowl-2026-prediction
Rank: #24
Source: https://www.kaggle.com/c/nfl-big-data-bowl-2026-prediction/writeups/public-23rd-solution-data-augmentation

This has been one of the most interesting competitions I’ve joined so far. I entered halfway through with only a month to work on the problem, and I wasn't very familar with ST-transformers. It was a real challenge, but also incredibly exciting. I learned a great deal thanks to the community.

I’ll cut to the chase and summarize my solution.

(I started from the public notebook like most people, so I’ll only highlight what I did differently and what I found helpful.)


---

# Model structure - ST-transformer play-level model

I applied the similar structure learned from paper **A Spatio-temporal Transformer for 3D Human Motion Prediction** [https://arxiv.org/abs/2004.08692](url), with slight tweak to fit this competition's data structure.

- Data input shape: (WINDOW, NUM_PLAYERS, FEATURES)
- Data output shape: (HORIZON, NUM_PLAYERS, 2)

Key model params as below:

    HIDDEN_DIM = 64
    NUM_HEADS_TEMPORAL = 8 
    NUM_HEADS_SPATIAL = 8
    NUM_LAYERS = 4
    DROPOUT = 0.05
    DIM_FEEDFORWARD = HIDDEN_DIM*4

I also added a few **edge biases** (sameteam_ij, distance_ij, closing_speed_ij) on top of the spatial attention output, which seems to help but not by much.

---

# Features

    'x', 'y', 'o', 'dir',
    'player_height_feet', 'player_weight', 
    's', 'velocity_x', 'velocity_y', 'accel_x', 'accel_y',
    'ball_land_x', 'ball_land_y',
    'dist_from_right', 'distance_to_goal', 'play_direction_right',
    'ball_dx', 'ball_dy', 'distance_to_ball', 'ball_dir_x', 'ball_dir_y', 'ball_closing_speed',
    'speed_change', 'dir_change', 'vel_x_change', 'vel_y_change',
    'passer_dx', 'passer_dy', 'passer_distance', 'vel_to_passer_alignment', 
    'receiver_dx', 'receiver_dy', 'receiver_distance', 'vel_to_receiver_alignment', 
    'post_pass', # 0=pre-pass, 1=post-pass
    'frame_to_pass',  # 传球前为负值(倒计时到0) - for virtual pre&post pass cut augmentation
    'input_frame_count', 'output_frame_count',  # total frame count
        
I noticed that adding more features no longer improves performance beyond a certain point. Eventually, I removed most of the fancy features and mainly kept those related to basic relative distances and velocities. I also retained several features related to frame count and their relationship to the pass point. Since I’m doing data augmentation similar to sliding windows (explained below), I want the model to understand the overall context of the time frame.

Besides, **player_role** and **player_position** are embedded as 8-dim feature each and added to the features above.

---

# Data augmentation - virtual pre&post pass cut (**most important**)

I only used available 2023 data that is provided under the data sector of this competition, which contains only 14108 plays after excluding bad plays. More data need to be generated to help the model generalize.

**Original data:**



**Augmentation method 1 (move pass cut into the pre-pass):**



Agumentation method 2 (move pass cut into the post-pass):





Augmentation method 1 & 2 together give me in total around **200k** more samples. 

Augmentation method 1 is more important, because it add players of all roles into output targets, which is very helpful to make the model more robust.

The augmented data is then used for train together with original data, but with a lower loss weight of 0.5.

---

# TemporalHuber Loss function (very important)

I learnt TemporalHuber Loss function from the communty, this is very helpful. Big thanks! I tweaked it a little bit to make the training slightly more stable:

    err = pred - target
    abs_err = torch.abs(err)
    huber = torch.where(
            abs_err <= self.delta,
            0.5 * err * err,
            self.delta * (abs_err - 0.5 * self.delta)
    )
  
    if self.time_decay > 0:
        L = pred.size(-1)
        t = torch.arange(L, device=pred.device, dtype=pred.dtype)
        w = torch.exp(-self.time_decay * t).view(1, 1, L)
    else:
        w = 1.0

    masked_weighted_huber = huber * mask * w # ◀️
    mask_sum = mask.sum(dim=(1, 2)) + 1e-8 # ◀️
    main_loss = masked_weighted_huber.sum(dim=(1, 2)) / mask_sum  # ◀️

---

# Train-valid strategy

I split data by week. there are 18 weeks' data, upon which I created 6 folds:

    FOLDS = [(3,7,18),(1,10,17),(4,5,8),(2,11,14),(6,13,16),(9,12,15)]

I use two random seeds for each fold, so each training run gives me a RMSE score averaged over 12 models. Still, my CV and LB don’t align well. At the 0.001 level, CV often goes down while LB goes up. I only trust CV when there’s a clear improvement of at least 0.01.

My instinct is to trust the LB more, because the best CV scores usually appear after tens of epochs, which feels more like overfitting or a random lucky hit.

---

# Eventual submission

The eventual submission include **48 models** across **4 training trials**, each with **12 models** from different folds & seeds, combined using simple weighted mean. 4 Trials are mainly different in output horizons.
