# 29th Place Solution Handedness Augmentation Trick and Missing Values Insights

Competition: cmi-detect-behavior-with-sensor-data
Rank: #29
Source: https://www.kaggle.com/c/cmi-detect-behavior-with-sensor-data/writeups/29th-place-solution-handedness-augmentation-trick

Thanks to Kaggle and the host for the interesting competition. Also, special thanks to my teammates @tahaalshatiri @i2nfinit3y @chronoscop. This would've been impossible without them.

# Summary:
Our solution is a weighted ensemble of several 1D-CNN variants trained with different settings (losses, augmentations, etc.). We modeled IMU-only, IMU+TOF+THM, and IMU+TOF+THM with missing rotations separately. Standard and more advanced augmentations were used to boost performance.

# @i2nfinit3y contributions:
Started based on public notebook : https://www.kaggle.com/code/myso1987/cmi3-pyroch-baseline-model-add-aug-folds
Features used : acc (x/y/z)、rot (x/y/z)、linear_acc (x/y/z)、angular_vel (x/y/z)、angular_distance、acc_mag、rot_mag、acc_jerk、rot_jerk、acc_pow、rot_pow、linear_acc_mag、linear_acc_mag_jeark、angular_vel_mag 、angular_vel_mag_jerk、rot_angle、rot_angle_vel 

* For full features model, 3 conv branches have been used: acc、rot、thm+tof，and then input embeddings into 1d-cnn backbone.
* For imu-only model, A GRU module has been added to predict drift feature, and then concatenate the output with other embedding and input them into 1d-cnn backbone.
* Added a gate layer for tof branch, and then calculated binary_cross_entropy loss with 0 or 1. 
* Used the following augmentations:
Mixup & Cutmix (0.5 probability)
Jitter & Scale
THM+TOF features dropout
Add drifts into feature
TimeStretch and TimeShift from notebook --> Had big impact (https://www.kaggle.com/code/alejopaullier/cmi-sequence-data-augmentation)
* Used CrossEntropy loss + TripletLoss (used the embedding before classifier head to mine hard samples)
* EMA, SILU, warmup
Top single model imu only: cv 0.854 lb 0.835
Top single model all features: cv 0.880 lb 0.861

# @chronoscop part:
Removed subjects：'SUBJ_045235', 'SUBJ_019262'
* IMU features: acc (x/y/z)、rot (x/y/z)、linear_acc (x/y/z)、angular_vel (x/y/z) 'linear_acc_x', 'linear_acc_y', 'linear_acc_z','angular_vel_x', 'angular_vel_y', 'angular_vel_z','angular_distance','acc_global_x','acc_global_y','acc_global_z','angvel_mag' 
* THM/TOF part: Using statistical methods, the time-domain characteristics of each TOF sensor signal over a certain period of time are calculated, including the mean, standard deviation, extreme values, and distribution shape (skewness, kurtosis).
* For the full model, I extracted features from IMU (acc, rot) and TOF using separate conv branches, then fused them into a 1D-CNN backbone. The CNN output is fed into a bidirectional LSTM to model temporal dynamics, and attention pooling is applied to obtain a final representation for classifying the 18 gestures.
* Using SGKF 10 fold, AdamW, mixup, and random masking for augmentation. A warmup-cosine learning rate schedule with EMA (from epoch 10) and early stopping (patience=30), and Augment same as @i2nfinit3y.
* The main task classification loss uses the cross-entropy with label smoothing (achieved through soft labels), supplemented by the binary cross-entropy loss of the gated branch, weighted by 0.2.
* Augment_left_handed_sequence (described below) 
OOF cv                  public LB  private LB
0.8844 ± 0.002    0.853          0.848
Reference public notebook
https://www.kaggle.com/code/myso1987/cmi3-pyroch-baseline-model-add-aug-folds
https://www.kaggle.com/code/hideyukizushi/cmi25-imu-thm-tof-tf-bilstm-gru-attention-lb-75

# Augmentations:
This is contributed mainly by @tahaalshatiri.
- It was noticed that `acc_x` had a negative sign on the left hand, which should not have occurred since acceleration is defined relative to the device orientation.  
- This was found to be due to the sensor being placed near the **pinky** on the right hand and near the **thumb** on the left hand, making the setup asymmetrical.  
- `acc_x` was flipped, and rotation features were adjusted so that both hands shared the same orientation.  
- The corrected rotation was validated using the following visualization notebook:  
  https://www.kaggle.com/competitions/cmi-detect-behavior-with-sensor-data/discussion/583413  
- Additional sensors were flipped so that both readings approximately pointed toward the same location on the face.  
- This augmentation made it possible to change all left hand data to right and vice versa. Adding this to the teammates notebooks provided a boost of ~0.004 for the leaderborad scores.
- We tried also to convert all left to right and train a model, then all right to left and train a model, then create left and right versions of test dataset then use each model accordingly. This boosted the single models by ~0.004 but it didn't work well in ensemble.
- Also It was observed that rotational features were missing in ~50 sequences from the training set and in over 30 sequences from the test set.  
- Although the missing values initially seemed negligible, training a model **without any rotational features or derived features** for these ~30 sequences led to a performance gain of ~0.004.
- Also Two subjects were identified as having worn the sensors flipped upside down.  
- These subjects were removed from the training set, which also led to a measurable improvement.
- An idea that was explored but not fully realized (due to time constraints) was handling different sensor orientations explicitly.  
- The dataset included **50 gesture–orientation pairs**, each with ~160 sequences, suggesting that the splits were stratified by the organizers and likely mirrored in the test set.  

- These orientation pairs showed large differences in scores for the same gestures. However, this information could not be fully leveraged in the current solution.
- Another idea we thought of but didn't try is to train video classification models on the 3D animations here: https://www.kaggle.com/competitions/cmi-detect-behavior-with-sensor-data/discussion/583118
- Tried spectrograms but didn't work.

# @mohammad2012191 contributions:
- Unified the team cvs and prepared the ensembling models and fixed many bugs.
- Had some augmentation ideas like: training on seq and seq[::-1] with 2 different branches and losses then avg results. This surprisingly stabilized and smoothed the training significantly, though no real gains obtained from it.
- Tried some special target feature (I call it nearest neighbor target), where i pick the target of the most similar sample and use it as a feature for the current sample. Also tried averaging the nearest k target. It didn't work here.
- Tried using the orientation head as an auxiliary target, it boosted the cv but not the lb.
- Augmentation of adding random missing values in the training data to mimic missings in the test.
- Adding the real length of the wave (before padding/truncation) as a feature to the model (It didn't boosted the score).
- Trying some ideas to separate the phase whether it is transition or gesture and use only gesture for the models. Turns out it is not important and performance is equivalent with/without them.
- Thought about modeling the problem using the raw waves plotted on top of each other. I though this might offer some really good diversity for the ensembling + no need to handle missing values. Couldn't find enough time to implement.
- Use ensembling weights between imu only and imu+tof+thm based on the missings percentage in the sequence instead of global weights. Didn't worked well.


Our final solution is a 3 branches solution:
- For imu only 
- For imu+tof+thm
- For imu+tof+thm with missing rotation 
We convert all samples to right handedness, and use weighted ensemble based on cv. Our final ensemble cv is 0.895 and lb 0.879 with private 0.849 (which is our 3rd top private. The top private is 0.851, where is the same model but with avg left and right ckpts. Turns out it generalizes well to unseen data). 

**Now, we have included the open-source training scripts of @chronoscop and @i2nfinit3y in the project link.**
