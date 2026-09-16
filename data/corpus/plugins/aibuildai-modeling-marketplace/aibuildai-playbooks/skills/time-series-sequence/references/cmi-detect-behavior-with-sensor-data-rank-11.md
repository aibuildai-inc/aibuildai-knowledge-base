# 11th Place Solution

Competition: cmi-detect-behavior-with-sensor-data
Rank: #11
Source: https://www.kaggle.com/c/cmi-detect-behavior-with-sensor-data/writeups/11th-place-solution

Very lucky to have teamed with lightsource. My solution was decent, but my imu modeling was greatly underpowered and his was very strong. We teamed rather late so it was lucky that when we revealed our cards to each other we had good complimenting pieces. The main theme of our solution was optimized modeling and then a little bit of extra postprocessing sprinkled on top to get a decent boost. The main postprocessing insight is [here](https://www.kaggle.com/competitions/cmi-detect-behavior-with-sensor-data/discussion/603544). Our ensembling stayed relatively simple, just 50/50 averaging between our approaches. We teamed late enough that we used totally different code bases and just ran them separately and then averaged results. 



Plots analyzing how postprocessing works with different relaxations from the 2 gesture limit per subject orientation

# lightsource part

full code: [github link](https://github.com/l1ghtsource/cmi-detect-behavior-w-sensor-data)


## Features

**Used:**
- IMU: base (`acc_x/y/z`, `rot_w/x/y/z`)
- From public solutions: `acc_mag`, `rot_angle`, `acc_mag_jerk`, `rot_angle_vel`, `linear_acc_x/y/z`, `linear_acc_mag`, `linear_acc_mag_jerk`, `angular_vel_x/y/z`, `angular_distance`
- TOF: raw 320 features
- THM: raw 5 features

**Unsuccessful experiments:**
- IMU: 6d quat representations (https://arxiv.org/pdf/1812.07035), sliding-window statistics, integrals (except linear velocity), euler angles, gravity-based features, positional features
- TOF/THM: inter-sensor differences, statistics, center of mass, gradients
- Demographics: failed due to limited number of subjects

## Preprocessing

**Used:**
- No normalization (better than per-sequence scaling and StandardScaler)
- TOF: replaced -1 with 255, then divided by 255

**Unsuccessful experiments:**
- Filtering methods (firwin, wavelet, savgol, butterworth, median) degraded performance
- Kalman filtering was neutral
- Left-to-right handedness transformations were not useful (did not increase cv and lb, although for private it would probably be important)

## Modeling

**Two models:**
- IMU only
- IMU + TOF + THM (model selected at inference based on TOF availability)

**Branches:**
- acc: `acc_x/y/z`
- rot: `rot_w/x/y/z`
- fe1: hand-crafted features (jerks, velocities, angles, distance,
etc.)
- fe2: lag/lead diff, cumsum
- full: all IMU features
- thm: all THM features (only in IMU+TOF+THM model)
- tof1--tof5: TOF features from individual sensors (only in IMU+TOF+THM model)

**Extractors:**
- Public_SingleSensor - based on the most popular public model
- FilterNetFeatureExtractor (only IMU only) - modified from https://github.com/WhistleLabs/FilterNet
- ConvTran_SingleSensor_NoTranLol modified from https://github.com/Navidfoumani/ConvTran
- ResNet1D
- Public2_SingleSensor (only IMU only) - based on another public model that uses an extractor for each channel
- MultiResidualBiGRU - modified from https://www.kaggle.com/competitions/tlvmc-parkinsons-freezing-gait-prediction/discussion/416410

**Aggregation:**
- Branch level: concat (better than MHA, comparable to GAT, GCN worse)
- Extractor level: multihead attention

**Heads:**
- gesture
- sequence type
- orientation
- additional gesture heads directly over extractors

**Unsuccessful experiments:**
- Architectures: inceptiontime, efficientnet, harmamba, transformers (husformer, medformer, squeezeformer, timemil, etc.), moderntcn, wavenet, 2D/3D extractors for TOF
- Using spectrograms, scalograms, line plots
- Heads: gesture start prediction, full behavior mask, demographic features

## Augmentations

**Used:**
- mixup and variants (cutmix, channelmix, zebra)
- channel masking
- jitter

**Unsuccessful experiments:**
- time shift, stretch, warp
- rotations, left/right reflections
- low pass filter augmentation

## Training

**Used:**
- Optimizers: AdamW, MuonWithAuxAdam (slightly better in some cases)

**Unsuccessful experiments:**
- TorchJD for multi-task learning
- Metric learning components to improve the separation of similar classes, also did not provide any gain (arcface, tripletloss, ...)
- EMA


# Ryan part
full code: [github](https://github.com/ryanchesler/cmi)

## Features

Virtually the same as lightsource, mostly things from the public kernels. I tried many different processings of the features. Trying to spectrogram them, turn them into random squiggle plots like in [this paper](https://www.nature.com/articles/s41598-022-25108-2) but the regular features with gravity removed and derivatives seemed the most useful. I tried to convert things to world frame or a representation where the starting direction was always consistent, but didnt find any success with this. 

## Preprocessing

- Similar to lightsource, only normalizing with batchnorm as the first layer
- TOF: replaced -1 with 300, then divided by 255. Similar except for I marked -1 as even further away because I didnt want it confused between an object far away and something that was totally out of view. Dont think it really mattered

## Modeling
I had many different variations of my models. Some were imu only and full sensor only, but most of them functioned in both scenarios. Training them with heavy sensor dropout so they were robust to many sensors being nulled out or as imu only. 

My main conclusion with the modeling was that we really did not want depth and high feature interaction and we did not want methods that encouraged global patterns. One early experiment that was really telling to me was training a model that had a linear layer that learned feature interactions before applying any convolutions to the sequences. This fit so quickly and overfit immediately. We really wanted simple methods that would learn the best from every channel and grouping of channels individually. 

This landed me on a solution that was similar to lightsources and the public kernels that had multiple different branches that were being late and simply fused. My best model was a model with lots of branches but very shallow depth. It had a different branch with two conv1d layers of kernel size 3, 5, 9. And also the same for group convs so only groups of features got interactions. Then it also had branches of lstm, gru and a two layer transformer. All of these then fed into a conv1d layer that finally merged and interacted these features. I used a form of branch dropout and also some temporal and channel dropout in between every layer. Really trying to encourage it not to get dominated by any one branch and not to learn from a specific time step or feature. 

## Augmentations

I tried a ton of options but didnt really find anything that added beyond mixup. I ended up using very high amounts of this. The difference between alpha of 0.4 and 0.8 was quite large in terms of both loss and validation f1

I tried various forms of warping, rotations, scaling, noise, sensor drift, hand flipping, but I didnt really seem to find anything that boosted things. 

## Ensembling and Postprocessing
I think the main contribution I had to the team was my postprocessing. My modeling was ok, but lightsource's was significantly better on imu and my whole ensemble was approaching similar full performance but it was with brute force and many models being ensembled together. When we merged a simple average between our solutions gave us a huge boost, but when I ported over my postprocessing to our combined solution that was when we were able to take first for a while. The biggest thing was just taking advantage of the relationship between subject gesture orientation constraints. We saw from our training data that only two gestures were done per orientation at most and locking this in significantly helped our score. The second piece on top of this was making sure illegal pairings were never predicted. 21 orientation gesture combos were never done in our data so turning these off didnt have a huge effect but it was still a decent gain in reference to how close our leaderboard was. 

We got another big boost when we converting the probablities that we were averaging before with a conversion to ranking. This was rather simple. Just assigning 18/18 to the highest confidence class 17/18 to the next and 16/18 to the third and so on. It was surprising to me that this was better than giving the models some expressive power by giving probability to each class, but maybe this worked to calibrate predictions per model. I noticed that lightsources predictions were way more confident than mine. Mine were often extremely blurry decision boundaries, probably because of the extremely high mixup. 


