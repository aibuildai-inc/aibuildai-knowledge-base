# 19 place solution （shake TOKYO）

Competition: google-smartphone-decimeter-challenge
Rank: #19
Source: https://www.kaggle.com/c/google-smartphone-decimeter-challenge/discussion/262941

## Team solution
First of all, thank you very much for hosting this competition. Here we describe the team TOKYO's solution. 

Unfortunately, our solution went through a shake down （Public 8th Private 19th）. We guess one of the reasons is that we targeted downtown & tree areas too much, while highway areas consist the major part of the private dataset.

Our solution consists of several algorithms, some of them introduced by other competitors and made public, the others developed by ourselves.

[Imgur]

#### 1. GNSS-based localization   ([link](https://www.kaggle.com/minomonter/gnss-only-estimation?scriptVersionId=70161014))
The first module is a position estimation based on GNSS data provided in *_derived.csv. With some simple tricks as follow, we achived LB: 6.665 in public.
- Using cauchy loss in least-square optimization
- Introducing Elevation Mask

#### 2. Delta prediction  ([link](https://www.kaggle.com/dehokanta/shake-tokyo-delta-predicition?scriptVersionId=70516831))
Here we trained a model which estimates the difference between estimated position \\((lat^i_{est}, lng^i_{est})\\) and corresponding ground truth position \\((lat^i_{gt}, lng^i_{gt})\\), based on the provided train data. We introduced several features and used LightGBM for the estimation.

#### 3. Outlier rejection & Kalman smoothing
Based on the idea introduced [here](https://www.kaggle.com/dehokanta/baseline-post-processing-by-outlier-correction) by [@dehokanta](https://www.kaggle.com/dehokanta), we introduced a rule-based outlier detection algorithm. Through several experiments, we've found out that the estimation accuracy improved significantly by rejecting the following points:
- A point where the estimated velocity （= distance between the neighboring points） is higher than \\(45 [\rm{m/s}]\\)
- A point where the estimated acceleration （ = difference of velocity between the two neighboring points） is higher than \\(1.8 \times 9.81 [\rm{m/s^2}]\\)
After this, we applied the kalman smoothing introduced [here](https://www.kaggle.com/emaerthin/demonstration-of-the-kalman-filter) by [@emaerthin](https://www.kaggle.com/emaerthin).

#### 4. Phones mean
We used several types of phones mean method. Extended from the idea introduced [here](https://www.kaggle.com/t88take/gsdc-phones-mean-prediction), we used several types of phone means such as:
- Align all the points from different smartphones into one line
- Align all the points from different smartphones by introducing a weight based on how "zigzag" the estimated trajectories are

#### 5. Stop detection & filtering
As discussed in several discussions & notebooks, the noise during the vehicle's stop was apparently one of the major noise in the baseline estimation.
We developed stop detection algorithm using inertial measurements in a rule-based fashion.

For some collections which does not include sensor data, we estimated the stop state only from the position data.

#### 6. Snap-to-grid
We used snap-to-grid for tree area. For downtown area, we applied a different snap-to-grid method described below.

#### 7. Dynamic Time Warping based snap-to-grid  ([link](https://www.kaggle.com/minomonter/dynamic-time-warping-snap-to-grid?scriptVersionId=70166586))
To overcome some severe problems in simple snap-to-grid, we combined the algorithm with Dynamic Time Warping （DTW）. This algorithm significantly worked especially for the downtown area.

#### 8. Remove Samsung
We used a method introduced [here](https://www.kaggle.com/columbia2131/device-eda-interpolate-by-removing-device-en-ja) by the other team.

------------------------------------------------------

###  What we tried but didn't use in the final pipeline
- RTK based localization using RTKLIB
- delta prediction by MLP
- pseudo labeling for delta prediction
- some other filtering methods, e.g. SGFilter
- Factor graph optimization based on [this paper](https://arxiv.org/pdf/2004.10572.pdf)
- outlier prediction by ML
- stop detection by density

### Final submission is visualized here.([link](https://www.kaggle.com/dehokanta/shake-tokyo-visualization-of-final-submissions))
