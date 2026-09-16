# 35th place solution

Competition: cmi-detect-behavior-with-sensor-data
Rank: #35
Source: https://www.kaggle.com/c/cmi-detect-behavior-with-sensor-data/writeups/35th-place-solution

# Intro

Congrats to the winners, and thank you for the organizers for preparing such a good competition. I am really grateful to be able to participate in this competitions, where people can learn from each other, share ideas and resources. Some notebooks were incredibly instructive. Others like the many huge shakeup ensembles not so much haha

This is my second kaggle competition - I am a beginner. I am happy to have gotten 35th place, though I am a little frustrated on many things I missed. I am also happy to have been able to learn so much during this summer school holidays, though I would have wanted to dedicate more time, as well as more compute, to this competition.

Trained two models, one with IMU only data, and another one with all data (imu+tof+thm). Gestures were used as labels, I did not do gesture+orientation labels I dont know why I just didn't even look at orientation label :( 

# Feature engineering and preprocessing

Raw quaternion data was cleaned by first interpolating missing values using Spherical Linear Interpolation (Slerp) and then fixing sign-flips to ensure temporal continuity. -1 in TOF was filled with 300 since -1 means far away like the signal didn't reach back. 

I got IMU features from this model [https://www.kaggle.com/code/jiazhuang/cmi-imu-only-lstm](url), and only I added quartenion diff features to that set.

Handedness normalization: acc_x, rot_y, and rot_z had to be multiplied by -1 to be flipped. Also for TOF and THM, sensors 3 and 5 were swapped, and flipped vertically. Sensors 1,2,4 were just flipped horizontally. I think that is the right physical transformation to apply to the sensor data, I checked looking at the data and it also seemed to be the right transformation.

Subject specific corrections: 'SUBJ_045235', 'SUBJ_019262' both wore the device upside down, so I was thinking about removing them, but ended up just doing the right transformation and just setting the TOF 1 sensor to Nan, since looking up you cant transform to looking down, but for the other sensors - acc x, acc y, rot x and rot y were flipped by -1, like in handedness sensors (THM/TOF) 3 and 5 were swapped, and all (2,3,4,5) sensors were rotated 180 degrees, which is the same as just flipping vertically and horizontally. This also seemed like the right physical transformation, and the data seemed to agree too, so I did not remove those subjects.

The sequences with 'bad' phase column values were manually corrected. 

For the TOF/THH: I had a 2d CNN to extract patterns from the 8x8 tof images, then concatenated with mean, std, max and min of the TOF and THM for each corresponding sensor, like TOF i with THM i. 

# Model Architecture

Note: The full data model was built on top of the already decent IMU model, not from scratch, so many things are similar.

Both models first process IMU data through a multi branch 1d CNN:
`self.feature_groups = {
                'raw_acc': ['acc_x', 'acc_y', 'acc_z'], 'raw_rot': ['rot_w', 'rot_x', 'rot_y', 'rot_z'],
                'linear_acc': ['linear_acc_x', 'linear_acc_y', 'linear_acc_z', 'linear_acc_mag'],
                'angular_vel': ['angular_vel_x', 'angular_vel_y', 'angular_vel_z'],
                'derivatives': ['acc_mag', 'rot_angle', 'acc_mag_jerk', 'rot_angle_vel',
                                'linear_acc_mag_jerk', 'angular_distance'],
                'rot_diff': ['q_diff_w', 'q_diff_x', 'q_diff_y', 'q_diff_z']
            }`

For the full data model, a performer is used to fuse data from the 5 TOF and THM sensors at each time step. This branch tries to model the spatial relationships between the different sensors.

From the CNNs in IMU only, and from Performer with the full data model, are fed into two parallel grus, one transition gru, and one gesture gru, based on the 'phase' column of the training data, aux loss is used for phase prediction and to use during inference. 

The final hidden states from both grus are aggregated using mlp based attention, max pooling and mean pooling, to get a final vector for classification head.

# Training

Data augmentation: For IMU only model, Mixup, time scaling, time warping, time shifting were used (I forgot to try scaling!). This code for reference: [https://www.kaggle.com/code/alejopaullier/cmi-sequence-data-augmentation](url). For the full data model, only jitter/gaussian noise worked, though I did not try too many augmentations for TOF and THM.

Standard cross entropy loss, stratified group per subject kfold, I did 10 splits instead of 5 and got less variance between seeds, so tests could be a little more accurate I guess, though having said this, I should have kept my model more light weight until later in the competition to be able to run more tests. EMA helped too.

I have seen many people do like >200 epochs, I am not sure if I did it wrong, but I just went with the default untuned training and all params I saw in this notebook [https://www.kaggle.com/code/jiazhuang/cmi-imu-only-lstm](url), so I just had 20 patience and trained for 50 epochs with 0.001 lr, I don't really know really well how to improve the hyperparamter intuitively without just sort of bruteforce tuning.

# Inference

I just did logits average of fold models. Jitter TTA improved LB by 0.001, but it was so little I did not include it. 

# Things that did not work
- Adding dilation to the CNNs, like stacking dilated CNNs, or TCNs, none of that worked. Scaling the model like adding more layers or higher model dim nothing seemed to work.
- Also could not get MHA to work on my IMU only model, though transformer and later performer to TOF/THM did work.
- I also tried Hierarchical Attention for classes which were confused a lot like the eyebrow and eyelash pull, and neck cheeck pinch for IMU, so I tried implementing expert heads, like first classify into the group, and then an expert head to disambiguate between the two similar gestures. I tried very hard to push this idea in different ways but nothing seemed to work sadly.
- For better pooling only MLP attention worked.
- Stats for IMU data for my current architecture it was just not good
- More stats like skew and kurtosis for TOF
- Better optimizers, somehow Adam did better than AdamW, though I fear this might just be noise/overfitting to validation
- Demographic features, not even with bins worked. I only used demographics for handedness
- I experimented with using FiLM to let demographic data dynamically rescale and shift the network's internal features, to change the model's processing of the time-series based on subject-specific traits after the cnn.
- I couldnt get image models to do well either
- Focal loss did not yield an improvement nor worsened cv
- SWA also didnt help though I think I implemented it incorrectly
- Collapsing the non target class labels into one global one 
- Adding more branches in the IMU multibranch cnns, like one branch per feature, or all in one branch, nothing did better than the branch sets I showed previously
- More, less kernel sizes also did not help, the public notebook I guessed had already good ones
- Filling with Nans some TOF sensors as augmentation also did not work, and I am not sure why, since there are sequences where this happens..
- Filling with Nans some patches of the 8x8 TOF image also did not help
- Chaning the IMU augmentations, like doing time shift, and then jitter for example, did not help, I did all the augmentations mentioned before in parallel
- Any sort of filters like the butter filter shared in moth's augmentation notebook did not help
- I think I did PCA wrong, though I dont think it would have worked well
- I tried fine tuning during inference based on sequences with high confidence pseudo labels, but again I think I did it wrong or my model was not good enough. Glad it was a correct idea, since I have seen daiwakun using it in his shared solution.
- I thought the acc x flip by -1 was not perfectly accurate and just a rough estimate, so I thought about adding the negative acc x as a new feature
- 6d quartenion representations did not help me either, though I believe again it was more my fault than the idea being bad :/
- Ensembling did not really work for me, I think because my models were not too diverse/different. 


# Results
0.842 IMU only CV. 0.898 Full data 10 fold CV. Avg CV would be 0.870, which correlated pretty well to public LB 0.870. Sadly private LB was lower at 0.845, though I did see everyones LB did drop at least a little I probably overfitted on validation.

Thanks for reading :)
