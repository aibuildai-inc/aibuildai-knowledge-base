# 2nd Place Solution

Competition: cmi-detect-behavior-with-sensor-data
Rank: #2
Source: https://www.kaggle.com/c/cmi-detect-behavior-with-sensor-data/writeups/2nd-place-solution

First, thanks to the organizers and hosts for an interesting competition, and to the community for insightful discussions and notebooks.

# Overview
- If you watched the leaderboard, you may have noticed a late jump: I discovered a dataset trick four days before the deadline and boosted the score substantially via post-processing.
- Without post-processing, my model scores were: Public **0.865** / Private **0.858**.
- Below I first describe the model (data handling, features, architecture, training), then the post-processing that made the big difference.

# Model

**Handling Missingness**
- Training separate models gave better accuracy when certain sensors were missing.
- I trained four variants depending on whether IMU rotation was missing and whether THM/TOF were missing (rot present/absent × THM and TOF present/absent).

**Feature Engineering**

- IMU
 - acc(x/y/z)
 - quaternion 6D representation (to avoid discontinuities)
 - angular velocity(x/y/z)
 - linear acceleration(x/y/z)

- THM/TOF
 - Filled NaN and -1 with zero.
 - I did not use THM.

- Left-handed subjects
 - Multiply by −1: acc_x, linear_acc_x, angular velocity y and z, and components 0 and 4 of the quaternion 6D representation.
 - Swap tof_3 and tof_5, and horizontally flip the left–right direction of the 2D TOF grids.

- Subject-specific correction
 - In the training data, data from SUBJ_019262 and SUBJ_045235 were rotated by 180° around the z-axis.
 - I multiplied all channels by −1 except acc_z and linear_acc_z.
 - I could not find a reliable correction for their TOF, so I set TOF to NaN for these subjects.

**Architecture**

- Based on public notebooks: a Residual SE-CNN Block + Attention model.
- Modality-specific stems:
 - Independent 1D CNN branches for each of: acc, quaternion 6D, angular velocity, linear_acc, and each tof1~5; then concatenate.
- TOF processing:
 - Before 1D CNN branches, I use a 2D CNN on the TOF grids per frame, followed by mean pooling to get a per-step feature.
- Phase-aware Attention
 - Plain temporal attention can over-focus on long phases (e.g., long transitions to the target).
 - I add an auxiliary 3-class phase predictor at each time step:
  1. (Relaxes and) Moves hand to target location
  2. Hand at target location
  3. Performs gesture
 - I construct three attentions, one per phase, and weight each attention score by the corresponding phase probability. This extracts features specific to each phase instead of letting long phases dominate.

**Prediction Target**
- I predict a composite label:(Initial behavior ∈ {Relaxes&Moves / Moves}, orientation, gesture).
- This choice is important for the post-processing step.

**Mixup**
- Simple linear mixup, but phase-aligned to avoid mixing behaviors:
 - Split each sequence into the three phases above and mix within the same phase only.
 - Especially for “moves to target,” I align the end of the phase so the arrival at the target is synchronized before mixing.

**Training Details**
- Optimizer: Adam (lr=1e-3, wd=1e-4)
- Scheduler: Cosine Annealing
- Batch size: 32
- Epochs: 50
- Folds: 10
- Final model: ensemble of 3 variants (different depths/layers of 1DCNN)

**Online Pseudo-Labeling**
- With ~8k training sequences and ~3.5k test sequences, pseudo labels helped.
- The submission API predicts one sequence at a time, so I performed test-time updates: whenever a batch-sized group of test sequences accumulated, I fine-tuned one step with a small LR (5e-5) using pseudo labels.

# Post-Processing

**Key observation from train.csv:**
- There are 4 orientations × 18 gestures = 72 possible (orientation, gesture) pairs, but only 51 were actually present.
- For each subject, the dataset contains exactly those 51 pairs, each recorded with two possible initial behaviors (Relaxes&Moves / Moves), totaling 51 × 2 = 102 sequences per subject.
- Based on this, I trained the model to predict 102 classes (the composite label described above).

**Maximizing Conditional Joint Probability (with no-repeat constraint)**
- For each subject, I keep all predicted class probabilities across their sequences.
- When predicting the N-th sequence for that subject, instead of taking the argmax for the N-th item independently, I choose the class sequence of length N that maximizes the joint probability (sum of log-probabilities) subject to “no label is used twice.”
- Because labels are assigned jointly per subject under a no-repeat constraint, early high-confidence picks lock in their labels; when a later sample is only moderately confident in a label that’s already taken, an alternative is chosen to yield a higher joint log-likelihood.
- Computationally, this is an assignment problem solvable efficiently (e.g., Hungarian algorithm) using negative log-probabilities as costs.

# Scores
| Method | Public LB | Private LB |
| --- | ---: | ---: |
| No pseudo / No post-proc | 0.862 | 0.854 |
| With pseudo / No post-proc | 0.865 | 0.858 |
| No pseudo / With post-proc | 0.891 | 0.875 |
| With pseudo / With post-proc | 0.900 | 0.878 |

# Codes
[github](https://github.com/Yamato-Arai/kaggle-cmi-detect-behavior-2nd-place-solution)
