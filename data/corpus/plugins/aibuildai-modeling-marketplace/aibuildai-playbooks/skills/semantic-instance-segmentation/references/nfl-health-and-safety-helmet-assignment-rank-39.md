# 39th place solution

Competition: nfl-health-and-safety-helmet-assignment
Rank: #39
Source: https://www.kaggle.com/c/nfl-health-and-safety-helmet-assignment/discussion/285059

Thanks to the organizers of the competition that we've had a possibility to take part in 😃
## 1. Detection
No custom detector was used, only raw bboxes from test_baseline_helmets.csv.
## 2. Camera Mapping.
I've tried to locate the global camera position (x, y, z) and adjust its rotation (yaw, pitch, raw) by some kind of the Iterative Closest Point algorithm (ICP).
### 2.1. Rotation Angles.
First of all, we need to get initial rotation angles so the camera can take a look at the field of players (90°, 180°, 270°). It is made by computation the euclidean distance ([cdist](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.cdist.html)) between corresponding points ([linear_sum_assignment](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linear_sum_assignment.html)) from tracking 2d projections and detections point clouds after transformations (move, scale). The points to be matched are chosen by up-down-left-right points and considered as a main cluster.
[Point Cloud Transformation]
It will also help in the measuring the approximate distance between the camera and the actual game center (the position of a player who make a ball snap).
### 2.2. Position Coordinates (with ICP).
Here the camera position is defined by spherical coordinates (r, theta, fi) with *r* equal to the distance to a ball snap player. We select the position with less distance between point clouds too. 
[Sideline Spherical Coordinates]
For Endzone videos, I've had to choose between two positions because the same projections are made by different angles (0°, 180°, 90° and 270°, 0°, 180°, as far as I remember...).
[Endzone Spherical Coordinates]
### 2.3. Adjust both by ICP.
Here we choose the direction of the camera position and rotation shifts and "move" them until the distance stops increasing. Then we choose the direction once again, and if it fails, we stop the algorithm.
[57583_000082_Sideline_10]
[57583_000082_Endzone_24]
## 3. Tracking by sensor data.
The ICP algorithm is used too.

Sometimes, it is necessary to disable "outliers" - detections that doesn't belong to any tracking data but are connected to some missing points anyway. First of all, we try to disable a single outlier if it is. If it fails, we try to find a main cluster by border points as it was described earlier (up-down-left-bottom points). It helps us to disable all the points that don't belong to the main cluster.
[58094_002819_Sideline_340]
## 4. Tracking by IoU.
It is the simple tracking by Intersection-over-Union (IoU) of bounding boxes with the threshold of 0.35.

If someone is interested in technical details, here is my notebook: https://www.kaggle.com/thehemen/nfl-camera-tracking-39th-place.
