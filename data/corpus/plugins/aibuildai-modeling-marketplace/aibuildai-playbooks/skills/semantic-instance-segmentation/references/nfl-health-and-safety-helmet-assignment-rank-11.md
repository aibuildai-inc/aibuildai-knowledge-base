# 11th place solution

Competition: nfl-health-and-safety-helmet-assignment
Rank: #11
Source: https://www.kaggle.com/c/nfl-health-and-safety-helmet-assignment/discussion/285156

First of all, I would like to thank the host for organizing such an exciting competition. I am very glad to have won my first gold medal!

# Overview
My solution consists of the following five steps.

1. Helmet detection
2. Tracking & bbox interpolation
3. Team classification
4. Player mapping
5. Post process


# 1. Helmet detection
- YOLOv5l, Input size:1280x1280
- Training data: 9947 images in "images" directory
- Validation data: ~1000 images from 120 training videos
- Use TTA ("augment=True" of YOLOv5's function)

# 2. Tracking & bbox interpolation
I created my original tracking algorithm. Basically my algorithm uses the distance of two points (LT, RB) of bboxes between two frames, it also predicts next frame's bboxes utilizing the optical flow. The optical flow method was inspired by the previous competition's top solution. When implementing optical flow bbox prediction, I referenced [this GitHub](https://github.com/jguoaj/multi-object-tracking). The IoU between predicted bboxes and detected bboxes is used together with the bbox's distance.

When the tracking is lost on some bboxes, they are left for two frames as the following figure. As a result, the number of bboxes increase. This is aimed at reducing the number of bbox's false negatives. Since the tracking is processed two times (forward direction and reverse direction), bboxes are interpolated in the previous and next frames.


# 3. Team classification
I trained the team classification CNN which has output size of 2(teams) x len(gameKeys). As the training images, I cropped helmet's bboxes. Aiming to utilyze the color of the uniform, I cropped bboxes a slightly larger to bottom direction than the original bbox's size.

- MobileNet v2, Input size:160x160
- Loss function: Cross-entropy
- Augmentation: Holizontal flip, Shift, Scale, Rotate, Shear, Brightness, Contrast, Cutout
- Feature size: 1280

The trained CNN is used as the feature extracter (the last fully connected layer is removed). The extracted features are splitted to two clusters by the K-means. At this point, I don't know which one is the "H" team or "V" team. The team mapping is completed in the next "Player mapping" step.

# 4. Player mapping
My base mapping algorithm is [this great notebook](https://www.kaggle.com/its7171/nfl-baseline-simple-helmet-mapping). I added some improvement to it.
- Two-dimensional mapping (x and y)
- To weight the x-axis of the bbox coordinates, the y-axis was normalized to 0~0.3 instead of 0~1.
- Use the Hungarian algorithm to minimize the mapping distance
- The team mapping is used as the cost of distance (the mismatch of the team increases the distance).
- Two combinations of team assignment between the K-means result and H/V are tried, and the minimum distance's one is adopted.

# 5. Post process
In this step, the final player label of each bboxes are determined by merging the tracking result and the mapping result. Firstly, the mapping accuracy of each frame are predicted by the XGBoost model. As input features of the XGBoost model, the following were used.
- The number of detected bboxes
- The number of players from tracking.csv
- Frame number
- The absolute difference between this frame number and the nearest tracking frame number
- Endzone or Sideline
- Mean value of x(y)-coordinate of all the players from tracking.csv
- Standard deviation of x(y)-coordinate of all the players from tracking.csv

Secondly, for each tracking IDs, each player's total predicted mapping score are calculated and the highest player is assigned. If the same player is assigned in the same frame, the bbox with lower total score is replaced to the secondly highest player.
