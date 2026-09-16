# 23th place solution

Competition: nfl-health-and-safety-helmet-assignment
Rank: #23
Source: https://www.kaggle.com/c/nfl-health-and-safety-helmet-assignment/discussion/285285

Thanks the host for an interesting competition and congratulation to everyone who fight until the end of competition. 

My approach doesn't use custom helmet detection because I have no experience in Computer Vision. The main improvement is in Tracking-Helmet Mapping and Post Process. Here is my approach





## 1. Helmet filter

I ran a simple iou tracking algorithm first to get cluster id and cluster score of each helmet box.
Cluster score is equal to maximum confidence score of all boxs belong to a cluster. A cluster is valid if its size is greater than 3.  Cluster score is then used to remove box when number of boxes greater than tracking player numbers. To remove outlier helmet ( sideline helmet ), I simple keep only boxes which have (top >= 70) & (top <= 650).

I think there must be a better way to assign cluster score but couldn't find out during competition but found some potential ideas in public solution and I 'll try after

## 2. Tracking-Helmet Mapping

#### 2.1 1-stage
I did some EDA and found that the score of individual frame is much higher when number of boxes >= 18. So I ran Tracking-Helmet Mapping process to only frame that have number of boxs above a certain threshold (I set to 17). I choose candidate players set by keeping player whose x is in [L, R] and y is in [T, B] and use linear_sum_assignment to assign boxs and candidate players, then use cluster id to vote label for each box, duplicated label in each frame is set to np.nan  

#### 2.2 Find Camera Direction
After 1-stage of Tracking-Helmet Mapping, I calculate median of top and left value for each team's boxes and compare this to tracking coordinate to find camera direction

#### 2.3 2-stage
2-stage use camera direction and some fixed label in 1-stage. Non-duplicated label in 1-stage is fixed and used as constrains in linear_sum_assignment. After 2-stage, cluster id is used again to vote label and drop all duplicated label

## 3. Post Process     
I add addition boxes for 6 frames before and after frames sequence of a cluster. This work because ground truth box is overlap to addition box in some case, and also there is no penalty in evaluate metric when predicted box and ground truth box is too far from each other.     

##### Summary

**CV 0.787 (run for all train dataset)** 
**+ Endzone   0.792**
**+ Sideline    0.783**

**Public   0.756**
**Private  0.790**
