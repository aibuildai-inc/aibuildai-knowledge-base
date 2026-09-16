# 3rd place solution

Competition: indoor-location-navigation
Rank: #3
Source: https://www.kaggle.com/c/indoor-location-navigation/discussion/240161

Thanks to the hosts for the great competition, and thanks to everyone who shared insights on the forum!
I'm so pleased to become a GM :-) It was worth spending a huge amount of time on this competition!

I tried many many things, kept some of them in the process, gave up with others, but it's quite difficult to say which were good or not.

First I compute the theoretical path (path shape : deltaX/Y from a waypoint to next waypoint) :
- my distance between 2 waypoints was obtained by a glm model, fitted on more than 2000 features, calculated with the accelerometer + gyro data. My rmse on distance prediction between 2 waypoints is around 1.1m. I think it was a competition inside the competition, to get a good distance model!
- the direction is obtained by using only the magnetic field. I tried to use gyro info, but it didn't improve my prediction. I also tried some filters with no success. The direction is often quite bad, like 30 degrees of deviation, but I didn't find a way to improve this.

Then I have the WIFI model : 
I mainly used a closest neighbor model. I applied some filters and thresholds on both train and test data (approx. 20 hyperparameters). I used the difference between timestamp and last seen timestamp to give a weight for each WIFI source. Also I have weights for each WIFI source as hyperparameters. This gave me a lot of hyperparameters.
I noticed that when I tune the hyperparameters of my WIFI model, the result could change a lot. One model will work for some paths and not for some others. So I optimized the hyperparameters for each path, until the path estimated by WIFI, had the same shape as the theoretical path. I optimized randomly with simulated annealing. This process was quite time consuming... I let it run during several weeks!
I also added WIFI information by computing the whole path on the training data and calculating WIFI information at a shorter timestamp, which gave me other models to blend, but was less good than the model with WIFI information located at each waypoint. 
I then used these models, along with all public notebooks WIFI models, in a quick post processing : for each path, I give a weight to each WIFI model, and optimize these weights in order to have the best fit with the shape of the theoretical path.
At the end of this step, my score is around 4m.

Improving the theoretical path :
I calculated several improved paths : 
- Starting from the theoretical path, I change the overall path so it starts and finishes where my WIFI model starts and finishes.
- I try to find similar shapes on a part of the theoretical path, and the training waypoints. When 3 or more consecutive waypoints match exactly (or nearly) with existing waypoints, after an X/Y offset, I consider that they are correct. Then I compute the remaining of the path (forward and backward) using the theoretical path. It's a "local" snap to grid, as I just move a few waypoints to the grid.
- then I did forward / backward iterations : starting from the WIFI position, I move according the the deltaXY and angle computed, and if there is a wall, I find the most logical direction to avoid the wall. This works well inside a corridor, but it's difficult to make it enter a narrow corridor. To avoid getting in the wrong corridor, I average the position at each timestep with the WIFI position. After 2 forward / backward iterations, most of the time my model gets in the right corridor.

Finally there is my snap to grid method :
I used an automated waypoint generation.
The snap to grid is obtained by calculating a probability for all training waypoints, at each time step. For the time step i, I calculate the probability to change from any training waypoint, to any other training waypoints. That's a bit similar to Markov processes, that I discovered in a previous competition (ion competition).
The probability is a product of several probabilities : 
- probability 0 if the path goes across a wall or a shop, 
- probability linked to the distance to the WIFI model, 
- probability linked to the distance to the any of the improved paths described above, 
- probability linked to the distance from previous timestep (compared to the predicted distance used in the theoretical path)
- probability linked to the angle from the previous timestep (compared to the magnetic field information)
- penalty if the path from previous timestep, was already existing in the training set
- probability linked to the time leakage for first and last step

Multiplying the probability of all waypoints at timestep i-1, by the matrix of probability described above (size n*n, where n is the number of training waypoint), gives a new probability array at timestep i.
I do this step forward, backward (giving a new starting point), forward again and backward again.
My final position is the weighed average of all waypoints (I take the average of the last 2 steps).

This final solution is used as input for a next iteration, by feeding the WIFI model with the positions obtained. 
I also completed training data with test data, by assuming my test data XY prediction was exact. 

It was my first solo competition, it removes part of the fun I think... But I needed my solo gold metal and thought it was the good time!
