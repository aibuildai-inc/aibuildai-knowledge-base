# 21st Place Solution - Hungarian Algorithm, Frame Filling

Competition: nfl-health-and-safety-helmet-assignment
Rank: #21
Source: https://www.kaggle.com/c/nfl-health-and-safety-helmet-assignment/discussion/285065

A lot of thanks to organiser for organising this competition and my best wishes to all the winners. This was one of the most interesting competitions I took part on Kaggle. I am glad to get a solo silver here.

Briefly Summarising my 21st place solution:

## Step 1: Helmet Detection

Two models were used in final submission: YoloV5l and YoloV5x. Both were trained on 1280 image size.

## Step 2: Helmet Mapping 

### (Linear Sum Assignment Problem - Hungarian Algorithm)

**1. The helmet assignment problem for each frame was framed as linear sum assignment problem as follows:**

  INPUT: Players at the boundaries of the video frame, Angle of rotation
  
  OUTPUT: All players in the video frame

    a. Assume you have solved the problem for 4 players at the boundaries (LEFT, RIGHT, TOP, DOWN) of the video frame and you know the angle of rotation (theta). 
    b. Transform the ground positions of all the players (x-center, y-center) by rotating the positions by angle theta and then applying min-max scaling. 
    c. Scale the video coordinates with min-max approach
    d. Compute weighted euclidean distance matrix: 
        For Sideline (0.5*x-dist + 0.5*y-dist) 
        For Endzone (0.8*x-dist + 0.2*y-dist) 
    e. Solve the linear sum assignment problem with hungarian algorithm to get all the players
    
**2. Grid Search over all possible candidates for players at boundary and angles of rotation**

    Suppose N_VIDEO = no of helmets detected in video
    N_GROUND = 22
    OFFSET = N_GROUND-N_VIDEO
    
    a. A set of 4 players are said to be valid candidates if no of players in the rectangle enclosed by these players on the ground is N_VIDEO-1, N_VIDEO or N_VIDEO+1 
    b. Grid Search over all possible 4 players and angles of rotation allowed for views to get minimum distance with linear assignment approach 
    c. Angles of Rotation allowed for Endzone = (90,270) 
    d. Angles of Rotation allowed for Sideline = (-30,-15,0,15,30,150,165,180,195,210) 
    e. If N_VIDEO>22, reverse the problem -> pick 4 valid boxes from video and apply grid search for players at boundaries on ground 
    f. If OFFSET<=8 problem can be solved in less than 1000 iterations
    g. If OFFSET>8 pick top 8 candidates for each direction -> Sort the coordinates and pick the ones whose distance from nearest players are highest
    
        For example: If offset = 9 and coordinates of players at left are : [1,5,8,9,13,15,18,22,25]
                                -> Reject 9 as its distance from nearest player on left side is minimum. Then apply grid search on remaining 8
                                
## Step 3: Helmet Tracking (Deepsort with IOU)

1. The assignment part of public deepsort code was modified 
2. IOU matrix was computed for deepsort coordinates and yolo coordinates. Hungarian algorithm was applied for deepsort cluster assignment
3. As in public notebook, Most frequent label for each cluster was chosen 

## Step 4: Ensemble

1. Helmet Mapping and Tracking part was run for each model separately
2. For each player and frame, predictions were ensembled with following approch
    a. If model 1 has some assignment for player, add model 1 solution for the player 
    b. If model 1 has no assignment for player, add solution from model 2
3. Note that: This approach works best if both the models are equally good.

## Step 5: Postprocessing (Forward & Backward Filling)

1. If for some player, we dont have solution at frame t, search for solution in frames belonging to (t-5, t+5)
2. Add solution from nearest frame if found. This approach works as the competition metric doesn't give any penalty on false positives and in most cases, there are not much movement in 5 frames. 


## CV strategy
 There are around 15 impacts per video. So, around 90 impacts in public test. The competition metric is highly impacted by these bounding boxes. I decided to go with full CV approach due to these observations
*  Public best notebooks has around 0.6 LB but had only 0.45-0.48 CV
*  My CV-LB difference was much lesser. Many times LB<CV
*  A better CV submission was not always better LB submission.
                                                         
I keep 2 validation set and compute 2 different metrics for each of them.

Validation set 1: 20 videos kept aside while training yolo with yolo predictions
Validation set 2: all 120 videos with baseline predictions                       
Metric 1: Competition Metric 
Metric 2: Assignment accuracy without giving any special weight to impacts 
                                                         
**I choose some strategy or hyperparameter only if it improves assignment accuracy on validation set 1 and competition metric on validation set 2**
                                                         
## Final Results
                                                         
**CV 0.805 (Endzone 0.81, Sideline 0.79)**
**Public LB 0.849**
**Private LB 0.795**
