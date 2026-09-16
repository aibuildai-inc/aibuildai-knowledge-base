# 15th place solution - Dive into compute_f without Grid Generation + Simple 100% KNN Floor

Competition: indoor-location-navigation
Rank: #15
Source: https://www.kaggle.com/c/indoor-location-navigation/discussion/240773

Thank the host for such an interesting competition. And also thank my great teammates for working hard throughout the competition ( @ryotayoshinobu, @tubotubo, @columbia2131 ).
The key factor of this competition is post processing based on algorithms rather than modeling. Also, all of us like competitive programming, so there is no other team’s name but “Algorithm is All You Need” :)
 
Here is our solution.


**Quick summary**




This competition required a good amount of pre and post processing. Our final prediction was a blend of 3rd stage models trained with iterative pseudo labeling. Repeating the post processing was also important for our solution.

**Simple 100% Accurate Floor model**

- Consider RSSI as the number of occurrences of a BSSID.
- TF-IDF vectorization.
- KNN for each site.

Our KNN model performed 100% accuracy for both public and private dataset.


**Waypoint Model**

**Preprocess**
-   	Linear interpolation of waypoints based on WiFi timestamp.
-   	Fix malformed txt data @higepon ‘s notebook 
https://www.kaggle.com/higepon/how-to-fix-malformed-train-test-data
-   	Remove WiFi information which exists only in training datasets.
-   	Applying Kalman Filter to sensor data for getting a precise result of compute_f.



**Cross Validation**
-   	GroupKFold of path
-   	5 CV but n_splits=15 to get many combinations of path groups to enhance random seed averaging.



**1st stage training**
-   	LSTM based on @Kouki ‘s notebook 
https://www.kaggle.com/kokitanisaka/lstm-by-keras-with-unified-wi-fi-feats.
Hyper parameters were tuned with Optuna.
-   	Transformer-like Convolutional Encoder.
Embedding BSSID and RSSI, then Positional Encoding are added.
No Feed Forward Network.
Batch Normalization instead of Layer Normalization.
Multiplying attention instead of Adding.
Query of MultiHeadAttention is BSSID embeddings, Key and Value are RSSI embeddings.
Encoding process is like [Embedding -> Conv1d -> Multiply Attention -> Conv1d -> Multiply Attention -> … -> Dense]
-   	10 random seeds to learn multiple combinations of path groups.



**2nd stage training**
-   	Convolutional stacking to learn correlation between different models or different random seeds, which is equal to learning multiple path combinations.
-   	LightGBM to learn time series and relative position information.
Additional input features are:
1. n predictions before and after
2. Difference between n predictions
3. Rate of change from n predictions
4. Difference between the previous and next n predictions
5. Moving average
6. Moving variance
7. Relative position (compute_f, compute_rel_position)
8. Cumulative sum of relative positions
9. Difference between predicted values and cumulative sum of relative positions
10. Aggregate features for each path (mean,max,min,median,std,sum)
 


**3rd stage training**
-   	Hill Climbing to get weights which minimize loss.
-   	Ridge regression to suppress overfitting.
 
 
**Post process**
Post processing is very important to push up scores significantly, and also a fun part for competitive programmers. Here we describe detailed steps. Summary is described below figure.







**Snap to grid**
published by @robikscube
https://www.kaggle.com/robikscube/indoor-navigation-snap-to-grid-post-processing
The idea is to replace the predicted coordinates with the nearest waypoint of the training data. There are two positive effects:
  1) pushing the predicted waypoint into the hallway when it is inside an obstacle.
  2) enables us to predict the same point as the training data.
As you all know, there are a lot of overlapping waypoints between the training and the test data. So, selecting the coordinate from train waypoints is very effective. Since our team applied iterative post processing, we set threshold 6 in the first half post-processing parts to increase probability of snapping, and threshold 1.5 in the latter half to avoid strange snapping of already precise waypoints.

**Parallel movement by Snap to grid**
As mentioned above, Snap to grid snaps predictions to the nearest point. However, the shape of the snapped path will not keep its original shape, especially most of the path is predicted to be inside a wall because snap direction is unpredictable. Therefore, before doing the snap to grid, we consider parallel moving of path to direction which is likely to be snapped while maintaining original shape. We calculate the difference between the original coordinates and the coordinates after Snap to Grid, and regard the average values of difference dx and dy as the direction in which Snap is likely to occur. By adding dx and dy to the original coordinates, we were able to move the path parallelly to the direction that is likely to be snapped while maintaining the original shape of the path.


**Cost Minimization**
published by @saitodevel01
https://www.kaggle.com/saitodevel01/indoor-post-processing-by-cost-minimization
 
The idea is to minimize the difference between the distance to the predicted point and the distance calculated by the sensor data.
 
⊿X^ is calculated by compute_rel_positions in compute_f. Though the host's prediction of relative coordinates with compute_rel_positions function is reasonably accurate, its accuracy depends on the quality of raw sensor data. If you look carefully at the relative coordinates, you will notice that the path of relative coordinates has been scaled up and that there is a bias in the direction of rotation compared to the original path. Let’s check the precision of this function.






Blue line indicates training path, and red line is relative positions. It is clear that relative position is scaled up and rotated compared to red line, so we need to calibrate compute_f function to get a good relative position. Compute_f function uses sensor data, so we try to clean up raw sensor data by Kalman Filter. Using Kalman filtered sensor data produces nice relative positions. We can get similar scale relative positions similar to the original path, which leads to a good score by cost minimization.






Next, we tackle with the rotation problems. Here is an example.









It is obvious that the red lines are rotated some degree. We try to solve these rotation problems with minimization of Euclid distance. 

Let the predicted coordinates in a certain path after post-processing be A1, A2, ... An in order from the start point, and the predicted coordinates by compute_rel_positions be B1, B2 ... Bn.If there is no rotation, the movement amount delta_a of A1 → A2 and the movement amount delta_b of B1 → B2 should be roughly the same. However, as you can see from the visualization, compute_rel_positions will rotate by some degrees, so delta_a and delta_b do not match. Therefore, we tried brute force search for the rotation angle θ of delta_b which minimizes the difference between delta_a and delta_b.
The loss function is the sum of the distances between delta_a and delta_b.

loss (θ) = || (A2-A1) – (B2-B1) R (θ) || ^ 2 + || (A3-A2)-(B3-B2) R (θ) || ^ 2 +… + || (An-An-1) – (Bn-Bn-1) R (θ) || ^ 2

Where R (θ) is a rotation matrix that rotates the coordinates by θ degrees. For example, (B3 – B2) R (θ) represents B2 → B3 rotated by θ. Also, || X – Y || ^ 2 represents the square of the difference between the movements X and Y.In other words, the loss function represents the difference between the "current movement amount" and “the movement amount calculated by compute_rel_positions rotated by θ”.The smaller the value of the loss function, the closer the angle of B is to A.





We modified the submission path by compute_rel_positions in the post process pipeline, and modified the rotation angle of compute_rel_positions by submission at the end of the pipeline. As we repeated the post process pipeline, both submission path and compute_rel_positions enhanced each other.
 

 
The difference between original and modified compute_f are very clear.







 In addition, cost minimization has two important parameters, α and β. The higher the α, the more emphasize the current prediction, and the higher the β, the more emphasize the sensor data. Since we use Cost Minimization many times in our pipeline, β was attenuated with each successive use so that Cost Minimization would not be too influenced by the sensor data.




**Push to Hallway**
If predictions are located in an obstacle, we need to push them to the nearest waypoint. But when we just increase the threshold of Snap to Grid, waypoints are changed to strange waypoints since there are many hallways which don’t have waypoints. So, we tried to push the waypoints to the nearest hallway even if there were no waypoints.
Each pixel of the floor image was used to determine whether predictions are located in the hallway or an obstacle, and moved waypoints located in the obstacle to the nearest coordinates in the hallway. This has the great advantage of being able to modify the waypoints in obstacles which is far from training waypoints.





**Apply Start and End Points leakage**
published by @tomooinubushi
https://www.kaggle.com/tomooinubushi/postprocessing-based-on-leakage

This was beyond our analytical capabilities, so we only corrected the start and end waypoints as same as the public notebook.

**What didn’t work**
Time series RNN.
Input all WiFi and sensor data to NN.
each model for each site.
Pre training of all data and Transfer Learning.
Dijkstra, Warshall-Froyd, Band First Search as shortest path solving 
Map Matching
Curve to Curve
Grid Generation




Thank you for reading our solution. 
Any opinions are welcome!
