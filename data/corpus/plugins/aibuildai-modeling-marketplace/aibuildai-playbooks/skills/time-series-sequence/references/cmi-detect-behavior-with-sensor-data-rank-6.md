# 6th Place Solution

Competition: cmi-detect-behavior-with-sensor-data
Rank: #6
Source: https://www.kaggle.com/c/cmi-detect-behavior-with-sensor-data/writeups/6th-place-solution

First of all, I would like to thank the competition hosts and all the participants for their efforts and contributions. It’s inspiring to see such creativity and dedication. I am also thrilled to have placed in the prize-winning range. Below, I would like to share an overview of my solution.

---

## Brief Summary

The main points of my solution are as follows:
- Used a custom neural network with separate 1D-CNN branches, one for each sensor modality  
- Extracted the last 75 frames of each sequence as inputs to the NN  
- Estimated gesture segments with a simple U-Net and added them as additional inputs  
- Converted left-handed data to align with right-handed data  
- Identified subjects who wore devices upside down (rotated by 180° around the z-axis) and corrected their data  
- Trained two nearly identical NNs (differing only in input blocks) 50 times each on the full dataset with different random seeds, then averaged the outputs of all 100 NNs
- Did not apply any post-processing  

Here are the cross-validation (CV) and leaderboard (LB) scores. Cross-validation was performed using 10-fold StratifiedGroupKFold.

| Model | CV or full-train | Handle upside-down<br>devices in test | CV<br>(single seed) | CV<br>(5-seed average) | Public LB | Private LB |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | CV | False | **0.864**<br>(0.894, 0.834) | **0.874**<br>(0.905, 0.843) | **0.869** | **0.860** |
| 2 | CV | False | **0.862**<br>(0.892, 0.833) | **0.875**<br>(0.904, 0.846) | **0.874** | **0.856** |
| 1 + 2 | CV | False | - | **0.877**<br>(0.906, 0.849) | **0.873** | **0.858** |
| 1 + 2 | full-train | False | - | - | **0.871** | **0.860** |
| 1 + 2 | full-train | True | - | - | **0.871** | **0.873** |

The notebook for model 1 in the table is linked below.  
https://www.kaggle.com/code/rsakata/cmi3-6th-place-solution-partial

## Neural Network Architecture

Of the two gesture classification neural networks, the structure of one is shown below. The other differs only slightly in the input feature blocks.



Each input feature block (shown on the left) is processed with 1D-CNN and MLP layers. It was important to perform the first convolution in each 1D-CNN block as a grouped convolution by channel.  

Before the MLP layers, the 1D-CNN outputs are pooled along the time axis. Instead of simple global average pooling, pooling is done separately over the gesture and non-gesture segments, and the results are concatenated as the input to the MLP. To enable this, gesture segments were pre-estimated with a U-Net and added as inputs. (The U-Net structure was quite simple, with no special tricks, so I omit the details here.)

The final prediction is obtained by combining the outputs from all blocks. Following the problem setup, the model produces predictions using both all inputs and IMU-only inputs. Additionally, separate predictions are generated for each block.  

Cross-entropy loss is calculated for all predictions and used for backpropagation. The loss weights are: 1/4 each for the all-input and IMU-only predictions, and 1/12 for each of the six block-specific predictions.  

In addition to predicting gestures (18 classes), the network also predicts orientation (4 classes), added as an auxiliary loss weighted at 0.5 relative to the gesture loss. Non-target gestures were not compressed; the model was trained as an 18-class classification task, since early experiments showed this performed slightly better.

For inference on the test set, if more than 50% of TOF values were missing, the IMU-only prediction was used; otherwise, the all-input prediction was used.

I will provide explanations for several feature blocks in the figure.

#### derived_acc

This block consists of the three components of linear_acc and the three components of global_acc. linear_acc is acceleration with gravity removed, calculated the same way as in many public notebooks. global_acc is acceleration expressed in the global (world) coordinate system; see the discussion in the link below.  
https://www.kaggle.com/competitions/cmi-detect-behavior-with-sensor-data/discussion/583080


#### rotvec_diff

This is essentially equivalent to what was calculated as angular_vel in public notebooks. (The division by time is omitted since the values are later standardized.)

#### tof

Before temporal convolution, each of the five TOF sensors’ values was spatially convolved on an 8×8 grid. After three rounds of stride-2 max pooling, the spatial dimensions were fully collapsed, producing 32 channels per sensor (160 in total). The resulting feature maps were then processed with temporal convolution in the same way as the other blocks.

---

## Preprocessing

### Handling Left-Handed Subjects

One of the most important preprocessing steps was handling left-handed subjects.

For **acc** features, the fix was simple. Because these features are symmetric left–right, flipping the sign of the x component was enough to convert left-handed data into right-handed. This is clearly visible in the acc histograms below.



For **rot** features, things were more complicated, so I’ll start with **rotvec_diff**. Note that rotation around the x-axis is the same for both left- and right-handed subjects (like twisting a motorcycle throttle). In contrast, the y and z components had to be sign-flipped — a step that proved critical for improving accuracy for left-handed subjects.

At first glance, it might seem that flipping y and z would also be enough for **rot** values. However, rotation around the z-axis depends on the subject’s facing direction (unless all subjects performed gestures facing true north). So I first canceled out the angle relative to true north, then flipped rot_y and rot_z. After aligning the left- and right-handed histograms, the subjects’ facing direction was estimated at roughly 130° from north (likely southwest).  
The before-and-after histograms below confirm that the distributions matched after this transformation.





However, this preprocessing is only possible because all subjects faced the same direction during data collection. For real-world use, models shouldn’t depend on absolute orientation, so excluding **rot** (and **global_acc**) values entirely might be a better approach.

For sensor placement, I also mirrored some values: thm_3/tof_3 were swapped with thm_5/tof_5, and each TOF sensor’s v0–63 values were flipped left–right.

These steps resulted in a gesture classification accuracy of about 85% for left-handed subjects with IMU+THM+TOF in CV.

### Handling Upside-Down Devices

Another key finding was that two subjects wore the device rotated 180° around the z-axis — specifically, SUBJ_019262 and SUBJ_045235. The histograms below compare these subjects with the rest.



The fact that acc_x and acc_y distributions appeared flipped strongly suggests the devices were worn upside down. I corrected the sensor values as if the devices had been worn properly:

- For **acc**: flip the signs of x and y components  
- For **rotvec_diff**: flip the signs of x and y components (z returns to original after two flips)  
- For **rot**: add a 180° rotation around the z-axis, and flip the signs of x and y components  
- Swap thm_2 with thm_4, and tof_2 with tof_4  
- Swap thm_3 with thm_5, and tof_3 with tof_5  
- Rotate the v0–63 values of TOF sensors  

These corrections not only drastically improved accuracy for the two affected subjects (accuracy exceeded 90% with IMU+THM+TOF in CV) but also gave a small boost to other subjects.

To prepare for similar cases in the test set, I also trained a model to detect whether a sequence was recorded with a 180° device rotation. The model architecture was exactly the same as that of the gesture classification network, differing only in the number of output classes.

Since this didn’t improve the Public Score, I wasn’t sure whether to include it. In the end, I added it to one of my final submissions but not the other. It turned out to make a big difference on the Private Score (+0.013), which significantly improved my Private Leaderboard rank.

### Other Simple Preprocessing

Additionally, a few other simple preprocessing steps were applied:
- For **thm**: Values below 20 °C were treated as outliers and replaced with nulls (filled with 0 after standardization)  
- For **tof**: Values of –1 were replaced with 255  

---

## Others

- Optimizer: I used RAdamScheduleFree. In the final stage, I attempted to switch to AdamW and tune it, and although I was able to reach similar performance, it did not clearly surpass the previous results, so I continued using RAdamScheduleFree until the end.
- Data augmentation and mix-up: I could not get them to work effectively, so I did not adopt them.
- EMA: I also tried using EMA, which improved single CV results, but its effect disappeared when combined with seed averaging, so I ultimately decided not to use it.

---

Thank you for reading!
