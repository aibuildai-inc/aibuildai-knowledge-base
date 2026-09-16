# 2nd place solution

Competition: nfl-health-and-safety-helmet-assignment
Rank: #2
Source: https://www.kaggle.com/c/nfl-health-and-safety-helmet-assignment/discussion/285112

Congratulations to all the winners. Congratulations especially to K_mat. Your scores and solutions are really great.  
<br>

I would like to thank the Kaggle and host team for this interesting competition. I especially want to thank Rob. Your contribution to this competition is very significant!  
<br>

This is a brief summary of my solution.
<br>

## 1 Helmet detection
I trained YoloV5 with only supplemental photos.
The prediction accuracy for small helmets was not good, so I upsampled from1280 to 1664 and trained it.


## 2 Helmet clustering (team classification)
I performed 2-stage K-means clustering using the helmet images.
The first K-means clustering  extracts 20 representative colors from the set of each pixels of all helmet images in the entire video-frames.
In the second K-means, helmets are clustered using feature vectors which shows how many of each representative color the helmet have.
This model achieved 97% accuracy.
This model was improved to 98% accuracy by using the tracking information to ensure that the same players are on the same team in post-processing.
The result of this team clustering is used for player assignment and player tracking.

[helmet clustering]

## 3 Feature extraction (used as distance for matching)
In addition to the 2D information of up/down, left/right, team (Home team or Visitor team) information, player orientation, and the gap between the helmet and the sensor position were used as distances for mapping players.
### 3-1 Player orientation
I created a model that predicts the "o" in the tracking data and used it to penalizes if the angle is too far off when matching.
[Player Orientation Prediction]

Red: Predicted value (Home)
Orange: Predicted value (Visitor)
Light blue: Ground Truth (Home)
Dark blue: Ground Truth (Visitor)


### 3-2 Apparent gap between helmet and sensor position
For example, even with the same coordinates, the apparent helmet position changes depending on whether the user is standing or squatting.
I have created a model to predict this shift.
In advance, I made mapping Grand-Truth and helmet box to obtain the positional gap.
Then I made a model to predict this gap using this training data.

This image show the helmet box updated by gap.
You can see that the boxes of kneeling and fallen players have been corrected.
[updated boxes with the gap]

For these feature extractions, I used [CenterNet](https://github.com/xingyizhou/CenterNet), and I shared this code here.
https://www.kaggle.com/its7171/nfl-centernet

## 4 Coordinate transformation
I used lines on the ground to align the coordinates of the video image and the tracking.
I used CV2 to detect the lines.
[White line detection]

By extracting two lines from the extracted line and doing a little calculation, geometric coordinate transformation is possible.

Coordinate transformation1 (aspect ratio)

[Coordinate transformation]

Coordinate transformation2 (trapezoid correction)

[Coordinate transformation]
Coordinate transformation3 (rotation)

[Coordinate transformation]

## 5 Player assignment
Since there are often less than 22 players in the video, I adjusted the number of players by removing  the players on the top, bottom, left and right of the tracking information (by cutting off the tracking coordinates with a rectangle), and then solved it as a linear assignment problem.
There are many candidates for removing players, but I chose the candidate with the smallest minimum distance among all candidates.

## 6 Player tracking
Modern object tracking models use the closeness of image features as a metric for the distance between objects.
However, I thought this approach might not be very appropriate for this competition since all helmets are the same for the same team.
So I used an algorithm based on [SORT](https://github.com/abewley/sort).
However, I introduced the helmet image feature as a step-function to avoid tracking the same person across different teams.
The image features are reused from K-means team clustering.
