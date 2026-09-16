# 17th Place Solution

Competition: cmi-detect-behavior-with-sensor-data
Rank: #17
Source: https://www.kaggle.com/c/cmi-detect-behavior-with-sensor-data/writeups/17th-place-solution

Big congratulations for all winners and thanks to the organizers! We unfortunately couldn't leverage subject-wise boosting tricks that many teams found out. But we are happy seeing that we got a robust ensemble! Our solution is an equal blend of two pipelines:

---

# 1. Anil's Pipeline

## 1.1. Modelling

- **Mode 1: IMU Only** - Uses accelerometer and rotation features when THM/ToF sensors are unavailable
- **Mode 2: Full Sensor Suite** - Leverages all three sensor modalities (IMU + THM + ToF)

## 1.2. Features
**Accelerometer-Only Features**
- **Raw acceleration:** acc_x, acc_y, acc_z
- **Jerk (acceleration derivatives):** acc_jerk_x, acc_jerk_y, acc_jerk_z

**Rotation-Only Features**
- **6D rotation matrix differences:** rot_6d_diff_col1_x/y/z, rot_6d_diff_col2_x/y/z
- **Absolute 6D rotation differences:** abs_rot_6d_diff_col1_x/y/z, abs_rot_6d_diff_col2_x/y/z
- **Relative rotation between frames:** rot_rel_6d_col1_x/y/z, rot_rel_6d_col2_x/y/z, abs_rot_rel_6d_col1_x/y/z, abs_rot_rel_6d_col2_x/y/z

**ACC-ROT Fusion Features**
- **Gravity Estimation:** gravity_est_x/y/z, gravity_est_jerk_x/y/z
- **Linear Acceleration:** linear_acc_x/y/z, abs_linear_acc_x/y/z
- **Jerk Features:** jerk_x/y/z
- **Subject-Aligned Features:** acc_tangential/lateral/vertical, jerk_tangential/lateral/vertical
- **Tilt Calculation:** tilt, tilt_diff

## 1.3. Architecture


**NaN Handling Layers**
- **IMU/THM**: Learnable embedding vectors for missing values
- **ToF**: Single tunable parameter broadcasted to all missing pixels

**IMU Branch**
- Two residual CNN blocks with SE
- Kernel sizes: 3 and 5
- Output: 128 features per timestep
- Mode-specific dropout: 0.3 (Mode 1), 0.1 (Mode 2)

**THM Branch** 
- Two residual CNN blocks with SE
- Output: 64 features per timestep

**ToF Branch**
- Spatial attention
- Depthwise separable convolutions
- Residual connections
- Multi-scale feature extraction
- Input: (5, 8, 8) per timestep → Output: 128 features

**Three-Modal Attention**
- **Cross-modal attention** between IMU, ToF, and THM
- Bidirectional
- Residual connections

**BiLSTM + Attention**
- 128 hidden units, bidirectional (256 total)
- Attention pooling for sequence-level tasks
- Dropout: 0.4

**Multi-Task Aux Learning**
- **Gesture** (main): 20 classes
- **Orientation**: 4 classes  
- **Handedness**: 2 classes
- **Behavior** (per-timestep): 4 classes

---

# 2. Optimo's Pipeline

I am going to share how my final pipeline looks like and try to explain how I got there. I did a lot of ablation study in the end showing that a quite simple solution could be competitive.

## 2.1 Data normalization

The dataset in this competition is quite small so it seemed important to represent each sequence in a more standardized way.

- **tof alignement**: as I was using 2CNN for tof feature extraction (see architecture later) it felt important to have pixels across sensors to correspond. So following the scheme of the device I rotate 90° tof sensors 3 and 5.
- **left handed alignement**: I did exactly the same thing as describe by @tatamikenn [here](https://www.kaggle.com/competitions/cmi-detect-behavior-with-sensor-data/discussion/603566) but his drawing abilities are far better than mine so I won't compete here! Transformation consists in change sign of correct acc axis and rot axis, as well as switch sensors 3 and 5 (thm and tof) as well as flipping tof images. The idea is that when a right handed person bring is hand to his head/body, the acceleration is from right to left while it's the opposite for left handed person.
- **upside down alignements**: it became obvious when looking at my oof scores that something wrong was happening to "SUBJ_045235", "SUBJ_019262". After a lot of data visualisation I came to the conclusion that they were wearing the device upside down. When applying the corresponding transfos (switching sensors, negating some acc and rot axis) those two subjects' scores went back to normal. To make sure that my transfo was correct I trained an "upside down detector": randomly applying the transfo to all sequences and predicting whether it was used or not. The model was very good at it and correctly said that those two subjects were indeed upside down. Problem is you do not have a similar info as 'handedness' to know that someone is wearing the device wrong. So I tried a sub with first using the detector and applying the transfo if needed before making my prediction: got a similar score (it was the two digit LB era at that moment). I thought "the test set is clean and everybody is wearing their device the correct way", so I just flipped those two subjects once and for all and continue improving my CV. **BIG MISTAKE**: some private subjects are indeed wearing their device upside down and the detector worked well (LB: 863 - PLB: 857). I should have done it the correct way: do not allow my CV to improve  after altering the data myself, my solution should have included the detector in order to improve CV and I should have trusted it!
- **sequence length**: all sequence are right cropped/padded to 96 (got similar results 64 or 128)

## 2.2 Data augmentation

I did a lot of augmentations on the fly:
- Alignement aware MIXUP: I randomly average sequences after aligning the start of the gesture so that averaging made more sense
- random rotaion: when looking at the videos of the lady performing the gestures on Youtube, you can see that the device is not always perfectly worn. So I add random small rotation (+-30°) around Z-axis to quaternions.
- random stretching
- randomly ignore some points at the beginning of sequences

## 2.3 Losses

I did not feel very comfortable using demographics data as there are two few subjects to be meaningfull for training. Nevertheless I tried to incorporate as much knowledge as possible to my models by forcing them to predict as many thing as possible (different heads):
- 18  gestures
- binary target
- orientation
- location: a custom mix of gestures into 4 locations (leg - face - neck -  front)
- action: a custom mix of gestures into 5 actions (pull hair - pinch - scratch - write - other)
- is child

Location, action and child were probably useless.

## 2.4 Feature engineering

I did little feature engineering:
- acceleration: original accelerations, temporal diff 1, acc_mal_jerk, normalized position in sequence (0 to 1)
- quaternions: orig quaternions, rotation angle and angle velocity (x, y, z) and distances, angular velocity norm and 10 window rolling mean of it
- thm: original thm
- rof: original tof

## 2.5 Architecture

I ended up with a relatively simple architecture, I only used one single model that learns to ignore missing information. I did not have one specific model for IMU only and full sensors: I simply randomly mask the thm and tof feats during training.

First encode information at the time level
- independant conv1D for acc, quats and thm for time level encoding (with kernels 3-5-7) with N_OUT features for each
- 2D CNN with 5 channels transforming 5x8x8 images to N_OUT (no time series info shared)

Then each stream goes through a gating unit which decides the importance of each stream of sensors:
```
class GatingUnit_v2(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.gater = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # x: (batch, feat_dim, seq_len)
        pooled_x = self.pool(x).squeeze(-1)  # (batch, feat_dim)
        gate = self.gater(pooled_x).unsqueeze(-1)  # (batch, 1, 1)
        gated_x = gate * x
        return gated_x, gate
``` 

Then I combine all feature by summing them which forces the model to share the same representation for each stream of data:
```
numerator = accel_features + orient_features + temp_features + tof_features
denominator = gate_accel + gate_orient + gate_temp + gate_tof

 combined_feats = numerator / (denominator + 1e-6)
```

Then I have an attention pooling mechanism to go from time series to series representation then basic MLP heads.


## 2.6 CV scores

Following the above pipeline I could reach avg CV 85.8 (IMU only 82.9 - all 88.7) with upside down subject flipped correctly.
Ensembling a few models with different training parameters could reach 86.4 CV.

My LB never went above 86.6 with my own models, I was never able to overfit public LB, not sure why.

Simply merging one 86.5 Anil's sub with one 86.5 single model of mine yielded LB 87.6 so I guess diversity in approaches was important here!

## 2.7 Few mistakes

The main reason I did not managed to improve my pipeline on the past month was because of stubbornness  and CV distillation. If using the OOF of a model to train a model from scratch following same CV scheme I could easily reach CV 0.88 (LB dropped significantly). It took me almost a month to accept that this was just pure leakage. If I decayed overtime the importance of OOF labels the CV went down but LB went up. I then tried 50 shades of leakage to reduce my CV and improve my LB like a crazy person.

I should have:
- accepted that distillation using OOF creates a powefull leakage
- compute my CV without allowing to arbitrarily put two subject upside down (I may try my upside down detector as late sub but I suspect a huge improvement on PLB)
- prob LB to exploit all available info as a real Kaggle champion should


Anyway! Thanks for reading this far, congratulations again to the winners and happy kaggling!
