# 30th place solution (vision transformer)

Competition: cmi-detect-behavior-with-sensor-data
Rank: #30
Source: https://www.kaggle.com/c/cmi-detect-behavior-with-sensor-data/writeups/30th-place-solution-vision-transformer

The key difference in my solution is the utilization of computer vision to train the sequence model.
# Importance factor
I found that there are four key factors that significantly impact the final result:
- Plus information: Post process using information from other sequence (Not used)
- Dataset bias: The test dataset only have right subject and the private dataset contains subjects wearing the device incorrectly, in an upside-down position. (Not used)
- Orientation: Use multi-task learning with orientation information. (Used)
- Model: Switching to xcit resulted in a significant improvement in CV performance. (Used)

# Model
I used a computer vision model because of its fast training speed and quick convergence (22 epochs, each taking about 40 seconds). The pipeline using 1DCNN + GRU is only used for the IMU-only model (it requires 120 epochs to converge, with each epoch taking about 110 seconds). The computer vision model I chose is Dino patch 8. Why Dino? Because among the vision transformer models, Dino offers faster convergence and better final results (likely due to good pretraining). Why patch 8? Because the TOF features are organized in an 8x8 matrix. I tried Dinov2 patch 14, but the results were not as good as Dino patch 8 (the score difference was about 1%). In the Dino family (Dino, Dinov2, Dinov3), only two models meet the patch 8 condition: ViT-S/8 and xcit_small_12_p8. Among them, xcit small performs significantly better than ViT small, but the training speed of ViT small is 1.5 times faster. 
## IMU only model

CV: 0.825
## IMU + THM + TOF model
| Model | CV  |
| --- | --- |
| xcit_small without softlabel | 0.875 |
| xcit_small withsoftlabel | 0.895 |

# Data augmentation
I use CutMix, MixUp, and RandomErasing. MixUp and CutMix contribute significantly to improving CV.
# Dataset
Padding method: 
- Training Augmentation: Randomly samples 80-100% of sequence length, maintaining temporal order
- Length Constraint: Crops to last 192 frames if sequence exceeds this limit
- Resampling: Interpolates sequences to exact sample_length (192 steps)

I discovered that the gesture performance segment only falls within the last 5 seconds (approximately 50 frames). However, when I cropped it to only the last 64 frames to make the model focus on the gesture segment, the accuracy decreased. This indicates that the transition segment also seems to play a role in providing additional useful information for the model.
Additionally, when padding, I intended to maintain the time interval between frames by inserting zeros at the beginning of the sequence. However, that method seemed ineffective. Ultimately, I performed padding by duplicating the elements to fill in the gaps around that element.

Correct wrong subject:
- SUBJ_019262: flip acc_y = - acc_y. rot_y = - rot_y.
- SUBJ_045235: flip acc_x = -acc_x. acc_y = - acc_y. rot_x = - rot_x. rot_y = - rot_y.

## Feature engineering 
Thanks to the authors of these two notebooks, I used features from them.
- https://www.kaggle.com/code/rktqwe/lb-0-77-linear-accel-tf-bilstm-gru-attention
- https://www.kaggle.com/code/nksusth/lb-0-78-quaternions-tf-bilstm-gru-attention
Features set:
[
        'acc_mag', 'acc_mag_jerk',
        'rot_angle', 'rot_angle_vel',
        'linear_acc_mag', 'linear_acc_mag_jerk',
        'angular_vel_x', 'angular_vel_y', 'angular_vel_z', 
        'angular_vel_mag', 'angular_vel_mag_jerk',
        'angular_distance',
        'roll', 'pitch', 'yaw'
]
To avoid having NaN elements between two elements in the sequence, I implemented the safe_diff_fillna_zero function and corrected the feature calculation method from the public notebook.
I used a Kalman filter to smooth the sequence and eliminate outliers, but the score improvement was not significant.
## IMU only model
### Computer vision model
Output shape: (1, 24, 192)
Normalize: not used
### 1DCNN + GRU
Output shape: (192, 17)
Normalize: not used
### 1DCNN + Transformer
Output shape: (192, 17)
Normalize: not used
## IMU + THM + TOF model
Output shape: (1, 352, 192)
Normalize: IMU not use. THM + TOF use
*The special aspect here is that I treat the feature dimension as the height (h) of the model and the sequence length (seq len) as the width (w) of the model. When the model shifts the patch window, it slides along the time dimension. If this is reversed, the model's accuracy decreases significantly.*
# Evaluation
I use StratifiedGroupKFold to split data (group by subject, stratified by gesture_handedness_height_cm_bin)
I use the last epoch as advised by @theoviel https://www.kaggle.com/competitions/cmi-detect-behavior-with-sensor-data/discussion/596787
# Softlabel
After training the k-fold models, I used the k models to predict the OOF (Out-Of-Fold) results. Then, I combined the predicted probabilities with the real labels at a ratio of 3:7 to obtain soft labels. Using these soft labels for training improved the CV score by 2.x%.
# Ensemle
I used the ensemble technique one model at a time. Since I made improvements but didn't have time to retrain the old model, the CV score of the old model is lower than that of the new model.
Result: Public=0.874. Private=0.849
