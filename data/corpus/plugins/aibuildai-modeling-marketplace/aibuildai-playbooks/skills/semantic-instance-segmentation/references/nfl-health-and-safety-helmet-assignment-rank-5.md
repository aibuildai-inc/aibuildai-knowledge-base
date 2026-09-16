# 5th Place Solution

Competition: nfl-health-and-safety-helmet-assignment
Rank: #5
Source: https://www.kaggle.com/c/nfl-health-and-safety-helmet-assignment/discussion/285286

First off thanks to the organizers / @robikscube for hosting this competition - it must have been the most fun I have had in a while, since I originally I just wanted to check some public notebooks (like @its7171 's awesome notebook) out and bail when I was happy but got sucked in :D. And congratulations to the winners, as some of your solutions are really a testament to how creative Kagglers can get.  The competition was also not too hardware intensive so it was very manageable resource-wise for a guy who is still stuck with his 1080Tis.

I'll be only verbal and graphically lacking since I'm exhausted from other commitments, and which is why I dropped the ball and slowed down in the last two weeks, plus I think some of my ideas overlap with other solutions anyway - but not to mention some unique differences in other solutions and more so perhaps the very unique top solution, which definitely blew my mind :D and is indeed a well deserved win on a different league.  I will try to augment the writing for this on the weekends when I get more time.

# Primary features
## Detector
- YOLOv5 (can't go wrong with this) pretrained from yolov5x6.pt, 1280x720 with standard augmentation as well as a 1472 version.  Both used hyp.scratch.yaml (not p6, which I didn't evaluate).  I used all the boxes in the videos beyond the ones given in `extra`.  In my final submission I used 118 videos and left only a pair for sanity checking [could have used all videos]
- WBF applied on the above and their horizontal flip TTA versions, using iou_thres=0.35, min_confidence=0.01.  Limited number of boxes to 22.
- I found that the detector performance greatly impacted my CV.  When I used the ideal boxes (from the ground truth) my score was around 0.958+ for the entire training set, which meant that detector performance was a bottleneck.
- I did not investigate but perhaps Scaled-YOLOv4 and YOLOR might have been better options given their metrics are supposedly higher. 
## Label assignment
- Euclidean distance minimization search by Hungarian assignment (use `scipy.linear_sum_assignment`, `lapsolver`, etc. and get their sum) including sweeping rotations, sweeping outlier tracker points as rectangular boundaries, permuting centering (mean vs min-max center).   Used min-max normalization on both images and tracker. 
 - After determining minimal distance for a set of parameters, determine label assignment from tracker to image again using Hungarian assignment via weighted Euclidean distance.  I found that the dy^2 was best at around 0.25 of the dx^2.  
- For subsequent frames, I also added a speedup to fix to one side via majority vote on the first 50 frames and limiting rotations and tracker outleir boundary search based on past frames.
- In addition, I added gravity features which scaled the cost matrix elements for the Hungarian label assignment, determined via individual discrete classifiers.  These included:
1. Digit matching -  I trained a multi-output classifier along the lines of the paper shared in discussions, which reached an accuracy of ~35+%. The classifier also had a side input branch that took an input vector of size 100 that one-hot any existing player digits that were available (since these were given to you), and added ~50% augmentation with generated digits on different colors. I used this classifier to generate digit features while the initial detector boxes were being generated, and in final application on Endzone videos only if one digit matches scale the image-tracker distance by 0.7; if two digits match scale the cost matrix element distance by 0.49.
2. Orientation matching - I trained an 8-class classifier with each class at 45 degree increments.  The angular difference between the classifier and the transformed tracker orientations was then inversely scaled with the corresponding cost matrix element.  For my final score I used 0.5 as the scaling factor, across an ensemble of 3 different orientation models.
3. Direction matching - Similar to the orientation model, but here I took +/-4 frames around the box of interest (similar to the previous NFL competition) and plugged them into a 2.5D CNN (replacing input layer) generating 8x angles for direction of movement.  Angular difference was then inversely scaled against the cost matrix element, and I used 0.7 as the scaling factor.  Anecdotally based on a submission check I think this improved my public/private score by +0.01.
4. Using past labels - scale the cost matrix element if the IOU of the current box and a previous frame box exceed a threshold for a given label.  This has somewhat of a dampened IOU tracker effect.  This helped in public, uncertain in private.

These significantly improved the pre-tracking validation/public scores (e.g. from 0.75 to 0.8+) - especially the orientation model - but barely moved any part of the post-tracker scores.  I suspect the images were too noisy e.g. if the individual was occluded or the digit was not visible the classifiers should either yield a low confidence or a null value but this would have required significant annotation cleanup that I was not ready for.. Also, adding these significantly slowed down my pipeline and mine tended to run close to 8+ hours for the full submission.
## Tracking
- Standard deepsort seemed to just work here, and the parameters seemed to indicate that low max age, high iou is preferable. 
- I use my own dynamic programming with bitmask function to maximize the sum of IOUs between the DS cluster boxes and the YOLO boxes to find the best correspondence between DS-detector boxes. I guess I could have done this using Hungarian assignment too.
- I did some extra post-processing to reassign duplicate labels, which basically took the re-assigned duplicated label clusters, sorted the counts of their labels, and assigned those labels which were not duplicated.
- If duplicates remained, as well as NaN labels, I used the tracker info to look for a nearest neighbor that was unassigned and just used that.
- One more feature -in trying to fix id switching - I added a parameter to break long clusters that had two / bi-modal large label counts since there were instances where the post-tracking score was lower than the pre-tracking score, but that occasionally did not work.  Example a cluster could have labels {A,A,A,A,....B,B,B,B} and if the counts of A and B are significant and over a certain threshold, I tried to break it up heuristically.  Perhaps a better way would be to use a learned model here...

## Misc
One more thing - I had one submission which had 0.912 public / 0.911 private, and the only difference of that submission with my current private of 0.885 is the number of epochs - the 0.911 private was using a epoch=10 1280 detector model, while the 0.885 private was an epoch=34 1280 detector model! The scores were definitely highly variable within most of the top positions, and perhaps I should have evaluated more carefully since that seemed to indicate the detector model overfitted beyond 10+ epochs.

## Summary of importance:
1. Detector - most important, this made the most impact
2. Hungarian assignment on Euclidean distance - most important
3. Gravity features - moderately important (tbd, I need to evaluate further) (?)
4. Duplicate fixing in tracker - moderately important
5. DP w/ bitmask maxmizing IOU btw tracker and detector boxes - most important
6. Cluster breaking post-tracking - questionable (?)

# What did not work
Oh boy, could I have a Kaggle point for every one of these:
1. FairMOT (even with YOLOv5 replaced as backbone), SORT.  V-IOU tracker was competitive, but I didn't find the time unfortunately to port the code over.  I also tried adding the gravity features from the matching above into IOU tracker, but that didn't help. 
2. Adding homography / cv2.AffinePartial2D / additional ICP on top of the Hungarian distance minimization.
3. Performing tracking first then using the tracked clusters to assist matching directly.
4. fast-reid retraining.  This marginally helped, but slowed down the pipeline a lot since the models were larger than the default REID provided.
5. Performing a forward and reverse tracking on the same video and taking the intersection of clusters (as an ensembling mechanism) to break clusters i.e. those with id switching.
6. In fact, I'm not sure if the gravity features mentioned above helped that much, but I'll do a more in depth evaluation against the public/private scores when I get the chance.
7. Tuning Kalman Filter parameters in DeepSort, e.g. with the position/velocity parameters and noise covariance.  Didn't seem to help.
8. Matching acceleration/velocity.  Training a 2.5D CNN regressing between min/max values on this gave extremely poor results and not surprising since it's hard to tell with the camera moving around.

# What I didn't add, but should have and was relatively low hanging and most top competitors did this:
Team classifier as gravity features.  I suspect this would have helped, but I saw some examples that had similar colors so decided against it.
