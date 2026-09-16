# 3rd Place Solution

Competition: cmi-detect-behavior-with-sensor-data
Rank: #3
Source: https://www.kaggle.com/c/cmi-detect-behavior-with-sensor-data/writeups/3rd-place-solution

Thanks to all the participants, this was a really fun and challenging competition. We are very happy with the 3rd place finish :) 

Our work is a result of great team-work and synergy by all team members - @minerppdy,  @nazarov  @ren4yu & @theoviel - and everyone contributed to achieving 3rd place.

## Overview

Our solution is an ensemble of pipelines carefully tweaked by individual members of the team. As-is, the ensemble achieves **Public 0.882 - Private 0.864**.

We jumped to **Public 0.897 - Private 0.878 (3rd)** by limiting each subject to 2 predictions per (gesture, orientation) pair. We'll explain this post-processing at the end of this section.

<a href="https://ibb.co/VWmnV31s">[CMI-1]</a>

The team pipeline contains one branch for IMU-only sequences, and one branch for full-data sequences (that reuses some IMU-only models), as illustrated below.

<a href="https://ibb.co/tT91YsMd">[CMI-2]</a>

## Post-processing

As noted above, we applied a post-processing step that limits each subject to two predictions per (gesture, orientation) pair. This optimization is done online and uses a Hungarian method.
We keep a cache of model predictions for each (subject, orientation) bucket in the predict function. For each new sample to be classified, we append it to the list for that bucket; let the list length be N. In this case, we have `(N, num_classes)` prediction matrix, and create `(N, num_classes * 2)` cost matrix, and run `linear_sum_assignment`. Here `* 2` means we allow each class to be assigned twice. Then we return the class assigned to the last row (the new sample).
An important point is that we optimize from scratch for each sample arrival, ignoring previous results returned to the API. We were also aware that some (gesture, orientation) pairs do not exist but did not include that in the algorithm (we thought of it too late).

Post processing results vary a lot depending on sequence order, resulting in us finishing at worst 6th - although a luckier choice might have gotten us 2nd.

## Overall ideas
Here are some key things shared across our pipelines. Although implementations might differ a bit, they provided key boosts to all models.

- Handedness normalization: For samples with handedness = 1, we canonicalize features to a common side.
  -IMU: left-right flip using a quaternion rotation of -110° on the z-axis first, or by simply swapping signs.
  - THM: swap the 3rd and 5th channels
  - ToF: swap the 3rd and 5th channels. We also flipped some of the the 8×8 images although it was less important
- Padding and truncating on the left makes more sense. 
- Convention for quaternions is to have rot_w positive. This results in sequence discontinuity. Some of us smoothed the data to make training easier. To make this actually improve models, we added sign flip augmentation or symmetric blocks for quaternions : `rot_block(quat) + rot_block(-quat)`
- The users `SUBJ_019262` and `SUBJ_045235` did not wear the device correctly, resulting in reversed y_acc and x_acc and a 180-degree rotation of the quat and tof data. They were fixed during training. Although we trained an outlier detection model to probe LB for similar outliers, we did not find anything. Probably an oversight on our end since @rsakata reported +0.01 with a similar strategy. 
- Data augmentation was important, among the most useful we found:
  - Mixup
  - Tof dropout
  - Stretch & shift the sequence
  - Rotation using the quaternion representation
  - Cutmix of sequences
- Features shared publicly were very strong, so we used them as-is or used some variants. It was quite hard to find new things.

## Theo's pipeline

*Written by @theoviel*

### Transformers
I initially worked on a transformer-based model, by starting from [this really strong baseline](https://www.kaggle.com/code/jiazhuang/cmi-imu-only-lstm). 
After fixing the left/right handed issue, I reworked augmentations, tweaked the architecture, added auxiliary heads and everything I could find that would help my CV. I tweaked the architecture separately for IMU-only and full data.

Although I could come up with useful features (using spectrograms, highpass filtering for instance), transformer-based models blended quite poorly together, and I think were a bit over-engineered. 

### Some ideas
- Features are computed after applying augmentations, which made more sense from the physical standpoint
Tof features (mostly region aggregation ones) were computed inside the model for GPU acceleration, and I used a 3D-CNN as well
- I tweaked the SE-1D-CNN block from public kernels a bit but I'm not sure it really helped. Some models used pooling to reduce the stride while others did not. Some also used parallel 1D-CNN blocks with different kernel sizes. 
- I also tweaked different transformers / RNN layers for diversity (GRU, Deberta, squeezeformer, skip-connections)
- I added a gesture/transition mask head with BCE which is also used for pooling, since most of the signal happens after the transition.

### CNNs

@nazarov had a lot of success with feeding raw input into 2D CNNs. Although that did not really make sense to me because it meant feature interaction would happen late in the model.
It worked almost out of the box in my pipeline, by doing a simple resizing to the desired image size. The key was to be a bit more careful with feature selection (use fewer features), and use (most) features twice in different orders to alleviate the feature interaction issue.
I got rid of the gesture mask head, and used an orientation head instead, which was reused for the post-processing. I ended up having best results with maxvit and convnext-v2 backbones.

### Results

The CV of CNNs was similar to transformers, but they blended much better in the team ensemble, and were stronger on LB. In the end, the following models were used:

- IMU - CV 0.844
- IMU + TOF + THM - CV 0.897
- **Public 0.868 and Private 0.852**


## Yu4u's pipeline

*Written by @ren4yu*

### Overview

#### Model
I'm using two models: one with IMU+THM+ToF, and another IMU-only.
Both models have an efficientnet-like 1D CNN backbone.
For the IMU-only model, I also used a ModernBERT backbone.
Features are similar to those of the team

#### Augmentations
Temporal scaling, mixup, 2d blockdrop for tof, randomly drop 3rd or 5th channel of thm and tof, gaussian noise for imu features

#### Auxiliary tasks
Train with auxiliary heads to predict behavior and orientation.

#### Training
AdamW optimizer with cosine annealing, 128 epochs, batch size = 32, learning rate = 2e-3, weight decay = 0.2, and an exponential moving average (EMA) decay of 0.9995.

### Results

- IMU only models - CV: 0.8423
- IMU+THM+ToF model - CV: 0.8857
- **Public LB: 0.862, Private LB: 0.847**

## Leonid's pipeline

*Written by @nazarov*

### Summary

This is my first experience with image processing, and some aspects of the approach may seem naive, but it worked. The core idea was simple: if I can visually distinguish between gestures on a feature plot, why shouldn't a pre-trained image classification model be able to do the same?

I consulted Qwen-coder for recommendations on suitable models and preprocessing techniques. The resulting pipeline is straightforward. 
The input data, which has the shape `(n_timestamps, n_features)`, was first resized to `(image_size, image_size)`. To make it compatible with models that expect three-channel images, the single-channel input was repeated three times, resulting in a final shape of `(3, image_size, image_size)`. This processed input was then fed into a model with a modified last layer to suit the classification task.
I experimented with all popular models under 200 MB, and the EfficientNet family appeared to be the most efficient.
I used EfficientNetB0, B3, B5, V2_S and V2_M in the ensemble. All models were retrained on full data. 

### Other things that worked

- Separating groups of features by arrays of zeros.
- Loss function, that was the weighted sum of 3 losses: CE for 18 classes, CE for 9 classes (non-target gestures were combined into one class) and CE for 2 classes (target/non-target).

### Results

EfficientNetB5 with CV 0.8310 on imu-only and 0.8868 on all-sensors data  was the strongest. The full ensemble score was **CV / Public / Private = 0.8636 / 0.863 / 0.859**.
A subset of it was included in the team ensemble to maximize diversity and optimize the team CV.

## Minerppdy's pipeline

*Written by @minerppdy*

### Key ideas
- Features
  - Relative quaternion to the next step based on the local coordinate system.
  - Angular jerk/snap
  - Acc jerk/snap
  - Linear acc
- Augmentation 
  - Rotate the quaternion around the z-axis of the world coordinate system by random angles (from -60 to 60 degrees) to increase robustness against the different directions the trial subjects face.
  - Rotate both the accelerometer and quaternion data around the y-axis of the local coordinate system by small random angles (from -7 to 7 degrees) to increase robustness against slight variations in how the device is worn.

### Model

The architecture is illustrated below, key idea is to use multiple branches to deal with different input channels.

<a href="https://ibb.co/SXrZb67k">[model]</a>

### Results

- IMU only models - CV: 0.8283
- IMU+THM+ToF model - CV: 0.8850
- **Public LB: 0.859, Private LB: 0.848**

## --

*Thanks for reading!*
