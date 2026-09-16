# 13th Place Solution

Competition: cmi-detect-behavior-with-sensor-data
Rank: #13
Source: https://www.kaggle.com/c/cmi-detect-behavior-with-sensor-data/writeups/13th-place-solution

First and foremost, I would like to extend my gratitude to the organizing team and staff for planning such a wonderful competition. 
I also want to thank all the participants who shared numerous insightful findings.

---
## Overview
Since the evaluation metric was unstable and there was a risk of leaderboard shake, and because single-model inference was fairly lightweight, my main strategy was to develop a variety of models and combine them into a robust ensemble with hill climbing.

#### Model
- Triple Branch Model (IMU + ToF + THM)
    - **Learns long-term temporal dependencies** for each sensor modality using a **Mamba-based architecture.**
    - **Captures sensor interactions** at each timestep using a **Transformer Encoder**.
- Single Branch Model (IMU only)
    - Based on an **SE-Transformer architecture**, which yielded the better accuracy for IMU-only sequences.

#### Feature Engineering & Data Processing
- Handedness Correction: Flipped data axes for left-handed subjects to standardize the dataset.
- Device Orientation Correction: Corrected data from subjects who wore the device upside-down.
- Handling Missing Data: Generated a variety of acceleration-based features for sequences with missing rotation (rot) data.


### Final Score
- CV: **0.9027** (Full Model) / **0.8537** (IMU only Model)
- LB: **0.873**
- PB: **0.861**

---
## 1. Model
From the beginning, I focused on creating a diverse set of models for ensembling, considering the unstable metric and the short inference time of a single model. I used a Triple Branch Model when ToF/Thermal data was available and a Single Branch Model for IMU-only cases.

### 1.1. Triple Branch Model (Mamba + Transformer)
I developed this model inspired by recent research showcasing the effectiveness of Mamba-based architectures in Human Activity Recognition.

#### Bidirectional Mamba Block:
To capture the time-series features of each sensor (IMU, ToF, Thermal), I designed three independent encoder branches. Each encoder consists of Bidirectional Mamba Blocks. While many papers make only the State Space Model (SSM) part bidirectional, my experiments showed that making the entire Mamba block bidirectional resulted in better performance. An ablation study on this Mamba Block showed a +0.005 improvement in CV score.

#### Transformer Encoder for Sensor Fusion:
After capturing long-term dependencies with Mamba in each branch, a Transformer Encoder aggregates them. This focuses not on temporal dependencies, but on capturing inter-sensor interactions at each timestep.
Think of it as experts from IMU, ToF, and Thermal domains discussing their initial findings to form a consensus. An ablation study on this Transformer Encoder also showed a +0.005 improvement in CV score.



### 1.2. Single Branch Model (SE-Transformer)
For IMU-only data, a model based on an SE-Transformer block outperformed the Mamba-based one. This architecture applies a Squeeze-and-Excitation (SE) block after a Transformer Encoder to learn channel-wise interactions. Ablation studies demonstrated a +0.01 improvement from the Transformer Encoder and +0.005 from the SE block.



---
## 2. Feature Engineering / Data Processing

### 2.1. Feature Engineering
Basically, I used the following features:
- imu_cols: acc_x/y/z, linear_acc_x/y/z, angular_vel_x/y/z, angular_distance
- tof_cols: mean, std, max, min, mean_delta, centroid_x/y for each ToF sensor (1-5)
- thm_cols: raw value and delta for each Thermal sensor (1-5)

However, for sequences where rot data was missing, results using rot-related features (like linear_acc, angular_vel) became deteriorated.
For these cases, I switched to an acc-only feature set. This set included 1st and 2nd derivatives of acceleration and rotation proxy features mimicking cross-products of acceleration vectors.
These additional features significantly improved the accuracy on sequences with missing rot values.
- d_acc_x/y/z (Jerk)
- d2_acc_x/y/z
- rot_proxy_xy/yz/zx (e.g. rot_proxy_xy -> acc_x * d_acc_y - acc_y * d_acc_x)

### 2.2. Data processing
#### Left-handed Data Conversion:
To address the scarcity of data from left-handed subjects, I flipped the relevant axes to treat it as right-handed data.
This allows the model to focus on the gesture itself, regardless of handedness.
- IMU: Flipped the sign of acc_x-related features and angular_vel_y/z.
- ToF/THM: Swapped sensors 3 and 5.
- This process was also applied during inference.

#### Upside-down Data Correction:
As pointed out in the discussions, two subjects (SUBJ_019262, SUBJ_045235) likely wore the device upside-down.
I corrected their data for training.
- IMU: Flipped the sign of features related to the x and y axes.
- ToF/THM: Swapped sensors 2 and 4, and sensors 3 and 5, respectively.

---
## 3. Training
- Fold: StratifiedGroupKFold (n_splits=5)
- Loss Function: Soft Macro F1 Loss + Hard Margin Loss
    - The Hard Margin Loss was introduced to push the decision boundary for difficult-to-distinguish samples (e.g., "Eyelash" vs. "Eyebrow"), which was a notable issue in the confusion matrix.

- Learning rate: 7.5e-4
- Batch size: 64
- Optimizer: AdamW
- Scheduler: CosineAnnealingWarmRestarts
- EMA (Exponential Moving Average): 
    - By using EMA(ModelEmaV3), more stable training and improved generalization.
    - decay: 0.9993 
- Hard Mining: 
    - In a 2nd stage of training, I increased the sampling rate of hard-to-classify samples identified by the model from the 1st stage. This led to a slight improvement in performance.


---
## 4. Ensemble
I trained a large number of diverse models (using different feature sets, architectures, and random seeds). 
Then I used Hill Climbing to find the optimal combination of models and their weights that maximized the CV score. The final submission was an ensemble of approximately 20 models.
- I thought the hill climbing score had some risk of overfitting, so instead of relying only on the CV score,
  I also considered the variety of models, the balance of weights, and the LB score.
  As a result, this balanced choice turned out to be my best private submission.


---
## 5. Post-processing
Final day in the competition, I realized that top competitors might using a trick based on the constraint that each gesture appears exactly limited times within a subject's session...
With limited time for thorough validation, I opted for a safer approach: for each subject, if the predicted count for a gesture exceeded 8, I changed the prediction with the lowest logit to the class with the second-highest logit. 
This simple post-processing step alone improved my LB score by +0.003.

---
## 6. Other Ideas (Did Not Work)
### Modality Hallucination
Inspired by several researches (e.g., [INERTIAL HALLUCINATIONS](https://arxiv.org/pdf/2207.06789)), I explored a technique to mimic intermediate representations of ToF/Thermal data using only IMU data. 
It resulted in a slight CV improvement but was not significant.
### Spectrogram Features
This method worked very well for EEG data in the HMS competition, but it didn’t perform as well in this competition.
That might be because of the large differences in movement rhythms between individuals, or because the raw waveform values played a more important role.
### 2-stage Model for Hard Classes
I attempted to use a specialized model in a second stage to classify confusable gestures, but this approach was not successful.
