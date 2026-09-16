# [3rd place solution] YOLOv5 + DeepSort + ICP + Hungarian algorithm

Competition: nfl-health-and-safety-helmet-assignment
Rank: #3
Source: https://www.kaggle.com/c/nfl-health-and-safety-helmet-assignment/discussion/285076

Thank you to all the participants and organizers!  
Thanks especially to @its7171 and @robikscube for sharing their amazing notebooks! 

### Summary

My solution consists of the following stages.

- (Stage-1) Use Yolov5 to detect helmets.

- (Stage-2) Use deepsort to track the helmet.

- (Stage-3) Use ICP (Iterative Closest Point) and the Hungarian algorithm to assign helmet boxes to the tracking data for each frame.  

- (Stage-4) Divide helmets into two clusters using k-means based on the color of the helmet.

- (Stage-5) Use ICP (Iterative Closest Point) and the Hungarian algorithm to assign helmet boxes to the tracking data again, taking into account the results of deepsort and the helmet color information.  

- (Stage-6) Use the Hungarian algorithm to determine the final label.  

### (Stage-1) Use Yolov5 to detect helmets.

I trained yolov5l6 and yolov5x6 with an image size of 1280.  
After converting the videos to images, I trained more frames with impact.  
In order to avoid detecting sideline helmets, sideline helmets were not trained.  
During inference, yolov5l6 and yolov5x6 were used, and only horizontal flip was used for TTA.  
Since helmets with impact are often only partially visible, I had to set lower thresholds for confidence and iou.  

### (Stage-2) Use deepsort to track the helmet.  
I ran deepsort on the helmets detected by yolov5 to track the helmet.  
At this time, to reduce false positives, I deleted helmets that could not be tracked for frames longer than a certain length.  
I modified the deepsort code to return the features extracted by ReID and the confidence.  
Features are used in stage-4, and confidence is used in stage-3 and stage-5.  
I used the ReID's weights distributed in git as is.  

### (Stage-3) Use ICP and the Hungarian algorithm to assign helmet boxes to the tracking data for each frame.  
I used ICP and the Hungarian algorithm to assign the helmet boxes to the tracking data.  
ICP is used to adjust the position of the helmet boxes so that the total distance between the helmet boxes and the tracking data is minimized.  
At that time, each coordinate is normalized to 0~1 for both x and y.  
Then the Hungarian algorithm is used to assign each helmet's boxes to each tracking data.  
The labels are assigned by creating and solving a cost matrix based on the distance between each helmet's boxes and each tracking data.  

In order to adjust the position of the tracking data using ICP, I need to remove False Positive helmet boxes and tracking data that is not shown in the video.  
To remove tracking data that is not shown in the video, I removed the outermost tracking data one by one and repeated the ICP.  
To remove False Positive helmet boxes in the same way, I removed helmet boxes with a confidence lower than 0.5 one by one and repeated the ICP.  
Also, when the number of helmet boxes is greater than the tracking data, remove the outermost helmet boxes so that the number is equal.  
It's because most of the time, the helmets of the audience are detected.  
After repeating the ICP and then getting multiple results, the result with the smallest total distance between the helmet boxes and the tracking data is chosen.  
Since ICP is highly dependent on the initial values, I reversed the top, bottom, left, and right sides and ran it in multiple patterns.  
[Alternate Text]

### (Stage-4) Divide helmets into two clusters using k-means based on the color of the helmet  
Helmet boxes were divided into two clusters  based on the color of the helmet using k-means.  
To calculate the distance between each Helmet's boxes, I used the features calculated by the deepsort process.  
Combining this result with the result of stage-3, I can determine which team (H or V) each helmet belongs to.  
[Alternate Text]

### (Stage-5) Use ICP and the Hungarian algorithm to assign helmet boxes to the tracking data again, taking into account the results of deepsort and the helmet color information.
I ran the same process as stage-3.  
The difference from stage-3 is that it takes into account the results of deepsort and the color of the helmet.  
This is done by reducing the value of the cost matrix according to the results of deepsort and the color of the helmet.  
When reducing the value of the cost matrix based on the results of deepsort, reduce the distance to the tracking data that has the same label as the label that occurs frequently in the deep sort.  
This increases the probability that the same label will be assigned to helmets of the same deepsort.  
When reducing the value of the cost matrix based on the color of the helmet, I use the results of Stage-4 to reduce the distance to the tracking data for the same team that the helmet belongs to.  
This increases the probability that helmets of the same color will be assigned the same team label.  

### (Stage-6) Use the Hungarian algorithm to determine the final label  
I used the Hungarian algorithm again to assign the helmets to the tracking data.  
In this stage, the cost matrix is created based on the percentage of labels in the same deepsort helmets.  
Therefore, labels that appear frequently in the same deepsort helmets are assigned priority.  
Relatively successful allocation of tracking data, which is difficult to adjust in ICP, because the distance between helmet boxes and tracking data is not used in the cost matrix.

### Score
validation score (24 videos) : 0.9102

public LB : 0.910

private LB : 0.904
