# Delta x,y CNN + MLP network from 5th place solution

Competition: indoor-location-navigation
Rank: #5
Source: https://www.kaggle.com/c/indoor-location-navigation/discussion/240852

After reading all of the other solutions, I think the big thing I did differently was that I was able to get < 1.0 MAE error (averaged over x and y) in predicting the delta x and y for the path "legs" (moving from one waypoint to another).

<br />
The structure of the network I used was a single main CNN for the imu data, but I then combined that with a lot of other path data into a big neural network.
<br />

# The network

Here's the basic structure:

[CNN MLP]

Here's a description of all the inputs and outputs:

## Inputs

### IMU 12 x 100

I collected all the imu data (acc, gyro, mag, rot) for each axis (x, y, z) which is the "12". Then I binned that data and averaged into 100 points for each leg. If there were < 100 points for a leg, then the rest of the values were just 0.

Some legs had a lot more than 100 points, but I found 100 to be a good middle ground of not too much data for the CNN to handle, but also enough data for the steps to be clearly visible on the acceleration graph.

I also tried 200 and 50, and saw almost no difference at 200, but slightly worse performance at 50.


### site_id

I converted the uuid of the sites to an integer 1 - 24, and used that as the input to an embedding (experimented with sizes here, but mostly around ~15)

### floor_id

Same thing with floor, but that number was 1 - 139. Still around 15 values in the embedding

### time_of_day

I calculated the local time of day (24 hour clock) and also added that to an embedding - my thought was that during busy times the collector might walk slower than not busy times. I'm not 100% sure that it helped (it would be interesting to do more tests adding/subtracting ALL of these values)

### day_of_week

Same thing as time_of_day, but for day_of_week

### total_leg_time

The total time recorded to move from waypoint A to waypoint B

### total_acc_time

I noticed in the acceleration graph, that it was very common for the collector to pause right at the start or end of the leg - I guess they were probably tapping on the phone, or figuring out the direction to travel etc. So, I figured out when the acceleration went above or below a 66%/33% line, and called that the "start" time, and then figured that out for the end as well.

This is what I called the "acc time" - which is the time the collector was actually MOVING, and not just sitting holding the phone.

### device_id

Using the device data leakage I outlined here: https://www.kaggle.com/c/indoor-location-navigation/discussion/234543 I binned any device within 2% calibration values to be probably the same device, and assigned an id to each one, then put those ids into an embedding.

The idea was that different collectors would have different stride lengths, and that might be able to be classified in an embedding

### device_calibration

Same thought as device_id, but with the raw calibration values from the sensor data.

### pct_leg

I calculated what % of the path this leg was located at - so the first leg of a path would be 0%, and the last leg would be near 100%. Not sure if this helped, but it was easy to calculate and I figured it couldn't hurt... again, it would be interesting to do a study to see what actually mattered out of all of these features.

### previous_point_xy_time

I calculated the previous point of the path's x, y, and time values (global x,y and time not delta x,y,time) when I was attempting to join this network with the global x,y network I was building with the wifi values.

The idea was that I could do some pseudo-label training with this, but it never seemed to work that well.  I took this out of later versions.

### next_points_xy_time

Same as previous_point_xy_time but for the points after this leg

### rays

In order to try to classify if the point was in a narrow hallway or wide open space, I calculated the distance from the point to the nearest wall in 16 directions from the start point. (so small values mean narrow hallway, large values mean wide open space).

This also served as a form of pseudo labeling that got better as my test predictions got better

### rot3_mid

In investigating the graphs, I saw that the rot3 (z value) was the most important imu value for determining direction of travel, so I grabbed the middle rot3 value from the path and added that as a standalone feature.

### before_waypoints

I added the delta x,y (or absolute x,y depending on which version of the network) for either the entire path, or just the last 5 points (again, based on the network version). I'm not sure if this helped a lot either, more experimenting is needed.

### after_waypoints

Same thing as before_waypoints, but for the waypoints that come after the leg in question.

### floor_waypoints

I included the nearest 20 waypoints from the training set (delta x,y from the leg starting point). The idea is that for most points (80% - 90%), the leg would go to one of these waypoints


## Outputs

### delta x

The distance (m) that the person traveled in the x direction (+-)

### delta y

The distance (m) that the person traveled in the y direction (+-)

### distance

The total distance traveled (m) - only positive values

### angle

The angle from waypoint A to waypoint B, from 0 to 2 PI (it took a bit of work to calculate that; and I'm still not 100% sure it helped or was necessary)


## Training

I trained with either MSE or MAE, with either Adam or SGD, varying the learning rate by hand (I could have saved time using a learning rate annealing, but didn't get it setup correctly).

I actually found pretty good results by starting with MSE/Adam, and then switching to MAE/SGD 1/2 way through, and then sometimes switching back and forth several times. That seemed to "shake" the network out of several local minimums, and allowed me to keep training for longer.

I also sometimes used a custom loss, using MSE or MAE for delta x and delta y, but then calculating the output distance from the delta x and y, and comparing that against the input distance and also the output distance. 

## Data Augmentation

The only data augmentation I did was to add noise to the input (Gaussian noise), which let me train for longer with less dropout.  I found added noise produced better results than higher dropout to prevent overfitting in this case.

## Fine Tuning

I trained this network on all 24 sites, but then also fine tuned on every site individually, then averaged the results from the full 24 sites, and the fine tuned networks.

## Ensembling

I ended up making about a dozen of these networks with slightly different inputs, structure and hyper params, and averaged the outputs.

## Results

The results were < 1.0m MAE for the large multi-site network, with the fine tuning getting some sites to 0.6m or 0.7m MAE (averaged over delta x and delta y).

Some sites had a _really_ hard time getting below 2m or even 3m in one case however, so it would be interesting to go back and figure out why that was and what would make that specific site better.


# Overall

I think this is one of the better results for the delta x,y networks described by other teams, and probably the reason I was able to take that output and apply post-processing so successfully.

I also don't think I hit the limit of what could be calculated - by either ensembling another team's network (like one of the RNN networks), or by doing a better job at figuring out which of those features actually mattered and using only those.

Also, because some of those features get better as the test predictions get better (a form of pseudo labeling), the results should only get better by running it more times with the outputs from my or other team's results (I only ran the entire pipeline 3-4 times).
