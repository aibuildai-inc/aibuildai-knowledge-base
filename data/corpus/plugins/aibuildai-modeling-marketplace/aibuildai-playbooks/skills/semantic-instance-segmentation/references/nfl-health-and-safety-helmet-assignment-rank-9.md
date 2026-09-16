# 9th place solution

Competition: nfl-health-and-safety-helmet-assignment
Rank: #9
Source: https://www.kaggle.com/c/nfl-health-and-safety-helmet-assignment/discussion/284940

First of all, I would like to thank hosts for hosting such a nice competition! The problem was so interesting that I was absorbed to this competition for almost all 3 months.

Here I will show you how I tackled this problem. The solution pipeline consists of 4 parts.
1. YOLO-v5 helmet detection
2. Tracking<->bounding box homography matrix estimation
    1. Homography matrix estimation using shape context in frame 1
    2. Modify homography matrix in previous frame to get that of current frame.
3. First helmet assignment using estimated homography matrices
4. Second helmet assignment using deepsort and previously obtained homography matrices

# 1. YOLO v5 helmet detection
As you know, baseline helmet detections were provided by the host. At first I thought the performance of the detection was enough and I used them for a while. However, I noticed some helmets involving in impact were missing. So I tried training a YOLO detector to suppress this false negative. The training settings was:

* Training data: jpg images in "images" folder, with 1280px resolution
* Model: yolov5x
* Batch size: 16
* Epochs: 20 (fixed)
* Mixed precision: on
* Environment: AWS SageMaker p3.8xlarge
* No validation
* Other settings(learning rate, optimizer, etc): default

It was quite effective and the score jumped up by about 0.1 at that time. 
EDIT: I replaced detections in my final submission with baseline detections and did late submission. I got 0.803(private)/0.841(public), which indicates that YOLO improved the score by 0.057.

# 2.1 Initial frame homography matrix estimation using shape context
To match sensor data to bboxes in frame 1, I used shape context. Shape context is a histogram of relative position of points. It can be calculated for each points and can be used in point set registration. For details of shape context, please google it. The detailed procedure was:
1. Normalize coordinates of bboxes (confidence>0.25) and sensor data. As a normalization factor, I used np.median(np.sort(cdist(points,points),axis=1)[:,1]). This quantity means median distance between nearest players.
2. Scale and rotate sensor data coordinates by some factor(mentioned later).
3. Calculate shape context for each points in tracking data and centers of mass of bboxes.
4. Calculate cosine distance matrix between sensor descriptors and bbox descriptors.
5. Hungarian algorithm matching using scipy.optimize.linear_sum_assignment. At this point, the matching result is usually contaminated by wrongly assigned pairs.
6. Homography matrix calculation using cv2.findHomography. I did this calculation like iterative closest point algorithm. During first several iterations I used cv2.LMEDS to repair wrongly assigned pairs. If the cost function stopped improving, LMEDS was turned off and switched to normal homography estimation.
7. If the cost function stopped improving, stop iteration.

The picture below describes the flow from 2.3 to 2.7.


## Cost function
I evaluated fitness of homography transform using the following code. This cost function can be interpreted as RMS distance between bboxes and transformed tracking coordinates.
dist=cdist(homography_transformed_tracking_points,bbox_points)**2
row_ind,col_ind=linear_sum_assignment(dist)
score=np.sqrt(dist[row_ind,col_ind].sum()/len(row_ind))

## Brute force optimization of rotation/scaling factors
I optimized rotation/scaling factors used in 2.1.2. In other words, I did above procedure many times changing rotation/scaling factors and selected best factors. The optimization ranges are:
* Rotation angle: [0, 2pi] with resolution pi/500
* X scaling factor: {0.5,1.25, 3}
* Y scaling factor: {0.5,1.25, 3}

# 2.2 Modify homography transform in a preceding frame to get that of current frame
Since player positions of adjacent frames are similar, the homography matrix should be also similar. First, I applied homography transform in previous frame to the tracking data in current frame. Then I assigned players to bboxes(conf>0.25) in current frame like aforementioned way to get current homography matrix. To improve robustness of the algorithm, I tried homography matrices in 60 preceding frames and selected one with best cost. To deal with false positive of helmet detections, I had to add rows full of zero in distance matrix (ignore false positive bboxes). The number of ignored helmets was adaptively selected by looking at cost function. If increasing ignore number from n to n+1 helped decreasing cost function by 10, the increase was accepted.



The typical quality of obtained homography transform looks like this (57583_000082_Endzone.mp4):

Blue: transformed tracking coordinates, Orange: BBox centers of mass

# 3. First helmet assignment using estimated homography transforms
Using homography matrix and ignore numbers in all frames, I assigned player label to each bboxes. The procedure is:
1. Assign labels to helmets with confidence>0.25. To assign them, I used aforementioned cost function and ignore number.
2. If the number of assigned helmets is smaller than 22, the all remaining labels and helmets were matched using linear_sum_assignment.

# 4. Second helmet assignment using deepsort and previously obtained homography transforms
I ran deepsort to improve assignment. The basic procedure was almost the same as the notebook Rob posted.
1. Run deepsort using bboxes which were labeled in first helmet assignment.
2. Assign labels to output bboxes of deepsort. The homography matrices were those which were obtained in secion 2. Helmet ignore numbers were recalculated.
3. For each deepsort cluster, calculate the number of assigned labels for each players.
4. Try assigning the most voted label to each deepsort cluster. If there were label conflicts in a frame, only one cluster which was voted most was labeled. Remaining ones remain unlabeled here.
5. Assign second most voted labels to unlabeled deepsort_clusters
6. Repeat 4 and 5.
7. If there were unlabeled clusters after 100 iterations, remaining labels were assigned to them using linear_sum_assignment.

## Parameters of deepsort
ReID model was the same as Rob's notebook.

* MAX_DIST: 0.28743019816658505
* MIN_CONFIDENCE: 0.4857573798672442
* NMS_MAX_OVERLAP: 0.43969024230191284
* MAX_IOU_DISTANCE: 0.7320514748544789
* MAX_AGE: 2
* N_INIT: 1
* NN_BUDGET: 30

# What I couldn't make it work
* FairMOT, ByteTrack, Tracktor++
* YOLOX detector
* Exploiting speed/acceleration/orientation data
* Jersey number detection (did not try)
* Training ReID models using torchreid
* Camera matrix optimization to get initial assignment guess for frame 1
    * Actually it worked to some extent, but it tended to be trapped to local minima. This tendency became stronger in videos with a lot of sideline players(e.g. 57781_000252_Sideline.mp4). So I concluded that shape context was better.
