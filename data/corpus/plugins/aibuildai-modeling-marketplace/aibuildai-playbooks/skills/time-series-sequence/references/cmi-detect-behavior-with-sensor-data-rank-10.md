# 10th Place Solution

Competition: cmi-detect-behavior-with-sensor-data
Rank: #10
Source: https://www.kaggle.com/c/cmi-detect-behavior-with-sensor-data/writeups/10th-place-solution

First, I would like to express my gratitude for organizing this exciting competition. For me, this marks my first gold medal in nearly two years. Moreover, since I previously achieved 10th place at the same Child Mind Institute competition through shakeup, these results feel particularly meaningful to me.

-----

## Training Strategy

  * **Epochs & Optimizer**: 200 epochs with **AdamW** optimizer and **EMA** (decay=0.999), no learning rate scheduling.
  * **Validation Strategy**: **StratifiedGroupKFold** with `groups="subject"`, `k=5`. CV splits were generated using three different seed values: 0, 1, and 2.
  * **Experiment Management**:
      * For clearly underperforming experiments, only the initial seed (seed 0) was run. All other experiments were evaluated across all three seeds.
      * The F1 metric showed significant instability; in some cases, performance was high in seed=0 folds but low in seeds 1 and 2.
  * **Model Selection**:
      * Adopted the best epoch based on the average performance across all 15 experiments (5 folds x 3 seeds).
      * Additionally, if the F1 metric was unchanged but the loss improved, that model was selected.

-----

## Preprocessing

### Handling of NaN values

  * **Rotation (`rot`)**: Replaced missing values with unit quaternions.
  * **Thermal (`thm`)**:
      * For each sequence, if any sensor value was below 18 degrees, we replaced that sensor's corresponding value with NaN. This was done to mask potentially corrupted readings.
      * Calculated the row-wise mean of NaN values and used it for imputation.
      * For rows containing only NaN values, we assigned a global mean of `27.03`.

### Hand preference correction

Adjusted all sensor values to a right-handed orientation.

```python
# Imu
df['acc_x'] *= -1.0
df['rot_y'] *= -1.0
df['rot_z'] *= -1.0

# Thm
thm_3 = df['thm_3'].copy()
thm_5 = df['thm_5'].copy()
df['thm_5'] = thm_3
df['thm_3'] = thm_5

# Tof
tof_names = [f'tof_{i}' for i in range(200)]
tof = df[tof_names].values.copy()
tof = tof.reshape(-1, 5, 8, 8)
tof = tof[:, [0, 1, 4, 3, 2]]
tof = np.flip(tof, 3)
tof = tof.reshape(-1, 5 * 8 * 8)
df[tof_names] = tof.copy()
```

### Z-axis rotation

It was determined that **SUBJ\_019262** and **SUBJ\_045235** had their sensors rotated 180 degrees around the Z-axis. The following correction was applied to their data.

```python
# acc
df['acc_x'] *= -1.0
df['acc_y'] *= -1.0

# rot
w = df['rot_w'].copy()
x = df['rot_x'].copy()
y = df['rot_y'].copy()
z = df['rot_z'].copy()
df['rot_w'] = -z
df['rot_x'] = y
df['rot_y'] = -x
df['rot_z'] = w

# thm
thm_2 = df['thm_2'].copy()
thm_3 = df['thm_3'].copy()
thm_4 = df['thm_4'].copy()
thm_5 = df['thm_5'].copy()
df['thm_2'] = thm_4
df['thm_4'] = thm_2
df['thm_3'] = thm_5
df['thm_5'] = thm_3

# tof
tof_names = [f'tof_{i}' for i in range(200)]
tof = df[tof_names].values.copy()
tof = tof.reshape(-1, 5, 8, 8)
tof = np.rot90(tof, k=2, axes=(2, 3))
tof = tof[:, [0, 3, 4, 1, 2]]
tof = tof.reshape(-1, 5 * 8 * 8)
df[tof_names] = tof.copy()
```

-----

## Data Augmentation

Due to the limited dataset size, data augmentation was crucial for model performance.

### Time-series stretching

  * **Quaternions(rot)**:
      * **Problem**: Quaternions can represent the same orientation with inverted signs ($q$ and $-q$). This causes "inversion points" in sequences where interpolation-based stretching produces meaningless data.
      * **Solution**: We preprocessed sequences to ensure all pairwise inner products were positive, creating a smooth orientation path.
      * **Method**: Used **Slerp** or **Nearest** interpolation methods.
  * **Acceleration(acc)**:
      * **Problem**: Simple interpolation (e.g., stretching by 1.5x) incorrectly scales the integrated velocity and position derived from acceleration.
      * **Solution**: We implemented a correction to preserve the start and end coordinates after stretching.
      * **Procedure**:
        1.  Use the orientation (`rot`) data to separate each frame's acceleration into its **gravitational** and **linear** components.
        2.  Apply stretching only to the linear acceleration component, then divide it by the square of the stretching factor.
        3.  Recalculate the gravitational component using the stretched `rot` values.

### Rotation

  * Rotating the data to simulate wristband misalignment was highly effective.
  * We applied a random rotational bias to the entire sequence along the Y, X, and Z axes.
  * The optimal standard deviations for the bias were: `std_Y=10`, `std_X=7`, and `std_Z=0.1`.

### MIXUP

  * **MIXUP** proved to be an extremely effective technique.
  * **Implementation**:
      * We applied MIXUP *after* feature engineering.
      * To prevent overfitting and improve performance over longer training, we applied MIXUP in **multiple stages**: `z = Mixup(Mixup(Mixup(x)))`.
      * To prevent excessive averaging, the original data `x` was concatenated with the mixed data `Mixup(x)` along the batch dimension.
  * **Parameters**: The first stage used `beta=0.5`, while the second and third stages used `beta=16`.
  * We also experimented with mixing three or more data samples by sampling from a Dirichlet distribution, but the multi-stage approach yielded better results.

-----

## Model Architecture

  * **Input**: The model processes input sequences of **128 frames**.
  * **Receptive Field**: Long-term temporal dependencies were not required for this task. The model's receptive field spanned **15 frames** (\~1.5 seconds), which was sufficient. This was achieved with 7 convolutional layers (kernel size 3) without pooling.
  * **Attention**: Features were weighted using **additive attention**, consolidated into a single representation, and then passed to a classifier.
  * **Two-Stage Model**:
      * **Stage 1**: An initial model estimates 4 posture classes and feeds these probability values to the second stage.
      * **Stage 2**: A second model inputs both the raw sensor data and the posture probabilities to predict the final 18-class and 9-class targets.
  * **Ensemble**: We created and used separate models: one for IMU-only data and one for all-sensor data.

-----

## Feature Engineering

  * **IMU (12 features)**:
      * Acceleration, acceleration norm, linear acceleration, linear acceleration norm, angular velocity, angular velocity norm.
  * **THM (10 features)**:
      * Temperature, temporal derivative of temperature.
  * **TOF (5 features)**:
      * Sensor-wise averages.

-----

## Reason for Significant Performance Improvement ("Shake Up")

  * We achieved a substantial improvement from **24th place (Public LB) to 10th place (Private LB)**.
  * **Primary Reason**: This improvement likely stems from the preprocessing step where we detected and corrected data from sensors that were flipped 180 degrees on the Z-axis.
  * **Score Impact**:
      * Public Leaderboard: **0.874 (no change)**
      * Private Leaderboard: **0.857 → 0.867 (0.01 improvement)**
  * We did not perform any private probing.
  * Given that the sensor-flip case occurred in 2 out of 82 subjects in the training data, the probability of it being present in at least one of the private test subjects was approximately 40%. Therefore, we consider this result to be fortunate.
