# 7th Place Solution

Competition: cmi-detect-behavior-with-sensor-data
Rank: #7
Source: https://www.kaggle.com/c/cmi-detect-behavior-with-sensor-data/writeups/7th-place-solution

First of all, I'd like to thank the organizers for hosting such an interesting competition. There were many different modeling options to explore, and I learned a great deal through trial and error.

My approach uses a **2D CNN**, inspired by a top solution from the HMS competition I previously participated in.

---
## **Overview**

* **2D CNN Approach**: I converted the time-series data into images and processed them using either EfficientNet or ConvNeXt.
* **Hierarchical Multi-task Learning**: The model has a main 9-class classification task and three auxiliary tasks (a detailed 18-class classification, orientation, and behavior).
    * The `behavior` is predicted as a time series from the backbone's intermediate feature maps. This prediction is then used as an additional feature for the main and other auxiliary classification tasks.
* **Exponential Moving Average (EMA)**: Used for more stable training.
* **Data Augmentation**: I used a variety of augmentations, including time-series specific ones (like Time Warping) and Mixup that was only applied between samples of the same orientation.
* **Post-processing**: A final step was applied to ensure that each subject in the test set had an equal number of assigned classes.

---
## **Preprocessing**

#### **IMU Data (12 dims → 41 dims)**
* **Base Features**: acc (3), rot (4), thm (5).
* **Engineered Features**:
    * Linear acceleration after removing gravity (calculated using quaternions).
    * Angular velocity and angular distance.
    * Jerk (the derivative of acceleration) and its magnitude.
    * Frequency domain features (power in the 0-1Hz, 1-5Hz, and 5-10Hz bands).
    * Detection of direction changes (using cosine similarity between consecutive frames).
* Corrected for handedness by inverting specific axes.
* Normalized all features using `StandardScaler`.

#### **TOF Data (320 dims)**
* The data structure was `(seq_len, 5*64=320)`.
* Corrected for sensor rotation and left/right differences.
* I found that arranging the TOF sensor data to match their physical proximity improved accuracy. The best order was:
    * TOF2 → TOF1 → TOF3 → TOF5 → TOF4

---
## **Data Transformation**

1.  **Crop**: Took the last 140 steps from each sequence.
2.  **Resample**: Resampled the 140 steps to 512 steps.
3.  **Resize**: Treated the resulting `(512 steps, 361 channels)` data as an image and resized it to `(512, 1024)` using `cv2.resize`.

---
## **Model**

#### **Architecture**
* **IMU only**: ConvNeXt-base (pre-trained on ImageNet).
* **IMU+THM+TOF**: EfficientNet-B5 (pre-trained on ImageNet).
* Used multi-scale feature extraction from stages -1, -2, and -3 of the backbone.
* The model has multi-head outputs:
    * **Main Cls Head** (9 classes)
    * **Auxiliary Cls Head** (18 classes)
    * **Orientation Head** (4 classes)
    * **Behavior Head** (4 classes x 64 steps, temporal)
* The behavior prediction was generated as a 64-step time series from an intermediate feature map. The resulting 4-class probability values for each step were added as features for the main and other aux task.

---
## **Others**

#### **Data Augmentation**
* **Time-series Augmentations**: Time Warping, Magnitude Warping, Jittering, Scaling, and Window Slicing.
* **Mixup**: Applied only between samples that shared the same orientation.
* **Rotation Data Dropout**: Randomly set quaternion features to zero with a 20% probability.
* **Variable-length Cropping during Training**: Used shorter sequences (70-200 steps) and longer sequences (120-700 steps) to handle variability.

#### **Training Configuration**
* **Optimizer**: AdamW (`lr=1e-3`, `weight_decay=1e-4`)
* **Batch Size**: 32
* **Scheduler**: Cosine Annealing with a 150-step warm-up.
* **EMA**: decay=0.99.

#### **Post-processing**
* I applied a post-processing step to equalize the number of predicted classes for each subject in the test set.
* I did not have time to implement a more complex version that also considered `(subject, orientation)`.
* The score improvement from this post-processing was **+0.004 CV**, **+0.01 Public LB**, and **+0.005 Private LB**.

|                     | CV    | Public | Private |
|---------------------|-------|--------|---------|
| Without postprocess (ensemble) | 0.884 | 0.872  | 0.864   |
| With postprocess (ensemble)    | 0.888 | 0.882  | 0.869   |

* The CV score is calculated excluding `SUBJ_045235` and `SUBJ_019262`.
