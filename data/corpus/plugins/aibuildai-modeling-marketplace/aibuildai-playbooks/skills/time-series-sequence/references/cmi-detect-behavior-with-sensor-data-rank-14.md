# 14th Place Solution

Competition: cmi-detect-behavior-with-sensor-data
Rank: #14
Source: https://www.kaggle.com/c/cmi-detect-behavior-with-sensor-data/writeups/14th-place-solution

Many thanks to the organizers—this challenge offered a rich playground of modeling ideas and yielded plenty of hard-won lessons

**TL;DR**
I used subject-wise GroupStratifiedKFold CV (5 folds for analysis; 20-fold bagging for the final submission), careful handedness normalization (mirror left→right), and 180° correction for two reversed-worn subjects. The modeling stack combined WaveNet, ResNet, U-Net, and SqueezeFormer on 96-step sequences, trained with sample weights, time-series augmentations, and EMA weight smoothing. Because some sequences lacked rotation (quat) features, I trained four model variants (IMU-only vs IMU+THM+ToF; with/without accel-only). At inference, I gated examples to the appropriate variant when rotation features were missing. This gating yielded a +0.0015 improvement on the Private LB, which kept me in the gold zone.

**Pre-processing & normalization**
- **Handedness normalization**: mirrored **left-hand** recordings to **right-hand ** coordinates to unify kinematics.
- **Reversed-wear correction**: two subjects wore the device reversed; I applied a **180°** rotation to bring their frames in line.
- **Sequence windowing**: fixed **sequence length = 96** samples per example.
- **Missing rotation features** : some sequences lacked quaternions/derived rotation features (Na). This motivated the model variants below and the inference router.


**Cross-validation**
- **CV scheme:**  GroupStratifiedKFold by subject to preserve class proportions while preventing subject leakage.
- **Folds:** I ran experiments with 5 folds (reporting CV), then built the final ensemble via 20-fold bagging (diverse splits/seeds under the same subject-grouped principle).
- **Why this matters:** subjects have distinct motion signatures; mixing a subject across train/valid inflates CV. Grouped CV produced stable CV↔LB alignment and reduced overfitting risk.

- 
**Features**
I combined competitive public feature ideas with additional temporal/physical descriptors:
- Public baseline features (adopted and pruned from community notebooks)
- Temporal gradients
- Absolute values
- Quaternion-derived features


**Model family (multi-architecture ensemble)**
All models consumed **96-step sequences** with identical train/valid splits.
- WaveNet
- ResNet
- U-Net
- SqueezeFormer


Four training variants (to handle missing rotations)
1. IMU-only (all IMU features).
2. IMU-only (accelerometer features only).
3. IMU + THM + ToF (all IMU features).
4. IMU(acc-only) + THM + ToF.

**Inference routing:** when rotation/quat-derived features were missing, I used **(2) or (4)** accordingly. Otherwise **(1) or (3).** This simple gate materially improved final reliability on mixed-modality test data.


**Training details**
- **Loss & weighting:** class imbalance addressed via sample weights
- **Augmentation (time series):** light jitter, time-shift, and scale perturbations; mirror/rotation ops conformed to physically plausible transforms.
- **EMA weight smoothing:** maintained an Exponential Moving Average of model parameters; EMA weights used for validation/inference to stabilize loss.


Many thanks to the competition organizers and the Kaggle community for resources and discussions
