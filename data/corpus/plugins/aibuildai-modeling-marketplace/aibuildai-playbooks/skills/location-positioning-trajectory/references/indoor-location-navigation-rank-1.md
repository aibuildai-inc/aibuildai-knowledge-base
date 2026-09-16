# 1st Place Solution - Track me if you can

Competition: indoor-location-navigation
Rank: #1
Source: https://www.kaggle.com/c/indoor-location-navigation/discussion/240176

We are excited to share our winning solution. It is extremely unusual to win on Kaggle with such a large margin without relying on obscure leaks. Our final solution was inspired by the tremendous sharing on the forum (especially [snap  to grid](https://www.kaggle.com/robikscube/indoor-navigation-snap-to-grid-post-processing) and [cost minimzation](https://www.kaggle.com/saitodevel01/indoor-post-processing-by-cost-minimization) were key insights). We combined the public ideas with unique modeling and optimization insights. The dataset in this challenge was very rich and allowed for impressive progress by many of the top teams until the last day, which made this my favorite competition so far.

We would also like to express our admiration for the team of @mamasinkgs. Your progress was very impressive throughout the challenge and your generous sharing on the forum inspired many of us to keep going. Your [prophecy](https://www.kaggle.com/c/indoor-location-navigation/discussion/235328#1288198) ("To be honest, I'm afraid Tom & dott team, who scores 6.2, is hiding a score now and show 0.x on the last day.") sounded like an untenable dream at the time, but after we got all the pieces together, we started to believe in it ourselves and we are proud that we actually lived up to your expectations.


## Summary of our approach
Like all top teams, we realized that this is essentially an optimization problem where you need to incorporate the relevant data modalities (predominantly WiFi + sensor data) at the trajectory level to achieve the best possible score. The key challenge was how to combine the predictions of the relevant data modalities with the discreteness of the prediction problem (about 85% of test predictions occur on X-Y locations seen in training).

Most top teams took the route of iteratively interleaving continuous optimization with discretization by snapping to the grid. We went for all-out discrete optimization instead, where we only considered the training waypoints, as well as the procedurally generated likely additional waypoints (see “Waypoint generation”) for each prediction. Discretizing the optimization allowed us to implement a customized Beam Search procedure which combined the penalties of all relevant predictions and allowed us to efficiently find the best match at the trajectory level.

We maintained a fixed holdout set (instead of cross validation), which was achieved by randomly selecting from the longer training trajectories. Our validation score was very well correlated with the private leaderboard score.

In what follows, we intend to focus on the insights that were not shared before on the forum.


## Base models
### WiFi data
We considered 3 modeling approaches for using WiFi data: NN, LightGBM and K-Nearest Neighbors. But only LightGBM and K-Nearest Neighbours were used in the final pipeline, having the highest accuracy. For both of them we assume that the floor is given.

The LightGBM models predict X and Y coordinates at the time of a WiFi observation. There are 100+ models, one per each site \* level. A particular boost in performance was observed by taking the maximum signal strength in a window of +- 8s each observation. Additionally magnetic and bluetooth data was used, but didn’t give a significant boost, resulting in a distance error of 7.25.

For the kNN model, all we had to do was to find a proper distance function between two WiFi observations, and compare WiFi observations at inference time with all WiFi observations seen during training. We settled on a distance function where you combine the average rssid strength difference for the shared devices with the fraction of shared devices. Less weight is given to device observations that are delayed (time difference between WiFi t1 and t2). A pointwise weighted kNN prediction got us to a distance error of 5.8

The main benefit of kNN over LightGBM is that it allows for a highly nonlinear penalty function which can incorporate more information about the layout of the floor without risking overfitting (we only have a handful of parameters in the distance function which are shared between all sites and floors).


### Sensor data
Sensor data was clearly the key information to reconstruct the trajectories in this competition. When using sensor data, we focused on predicting segments - pieces of trajectories from one waypoint to the next one. We ended up using 3 different sensor data models with different targets.

The first sensor model is targeted at predicting the relative movement of a segment, i.e. changes in absolute coordinates between two consequent waypoints. The model uses 9 inputs (acce, gyro and ahrs of all 3 axes) and has a relatively simple structure: Conv1d + GRU + Conv1d + head. The presence of GRU is unnecessary, similar results can be achieved when GRU is replaced with Conv1d, kernels sizes are set wider and dilations are added. The model has MAE = 1.04 on validation (averaged over the X and Y dimensions).

After analyzing the errors of the model, we saw that the predicted angle of the movement direction is far from perfect, though never off by more than pi/2 radians. That can indicate that the device was not properly calibrated.

[](https://www.kaggle.com/c/indoor-location-navigation/leaderboard)

Even when the direction was not properly calibrated, the sensor data still can describe the movement well, just in slightly rotated coordinate axes. To take advantage of that, we fitted the second sensor data model, where the target relative movement was expressed not in the original coordinates, but relative to the previous segment, as illustrated below:

[](https://www.kaggle.com/c/indoor-location-navigation/leaderboard)

The architecture of the NN remained the same, but this time we pass 2 consecutive segments (AB and BC) to the model. The NN is applied to both segments, after that the second segment (BC) is rotated so that the prediction of the first segment (AB) is along the first axis. The rotated vector BC is the output of the model.

This calibration issue also leads to a bias of the predicted distance between the visited waypoints - the relative movement model underestimates it (6.65 vs 6.91 on average in validation). The third model, fitted to predict the distance between the consecutive waypoints has a much smaller bias and, as the result, is better in predicting the traveled distance between waypoints (MAE of 0.67 vs 0.73). The RMSE of the distance model was 0.92. The architecture of the model remained the same, with only the head of the network changed accordingly.


## Data leaks
We are grateful to @chris62 for [sharing the data leaks publicly](https://www.kaggle.com/c/indoor-location-navigation/discussion/234543). At the time of the post, we were only aware of the time leak.

The device leak was very useful to identify periods in time where the sensor data was unreliable, since the errors are clearly time dependent. This enabled us to make the optimization less reliant on sensor predictions when we predict them to be noisy.

[](https://www.kaggle.com/c/indoor-location-navigation/leaderboard)

We realized that by combining the time and device leak, one could see that apparently different device ids are likely coming from the same phone. It turns out that the time between sensor observations (after filtering outliers) correlates very well with the device ids, which could enable you to group different device ids. In the end, we never used these fused device ids, but think that there could be more to be learned from our observation.

[](https://www.kaggle.com/c/indoor-location-navigation/leaderboard)

Our test floor predictions combined the time and device leak with WiFi distances. Sadly, we got 102 private floor predictions wrong (about 0.18 total score penalty). Trajectory ”862a4ac32755d252c6948424”, for example, should apparently be F5 instead of F4.

[](https://www.kaggle.com/c/indoor-location-navigation/leaderboard)


## Leaderboard probing
We quickly realized that probing was a viable strategy, since the test data set (626) was split by entire trajectories. After a couple of days of probing, we identified the 100 public test trajectories (1527 predictions in total). This was mostly useful for understanding the difference in distribution of the test data between the public and private set. The private set is clearly much harder, since it contains some notoriously hard trajectories (e.g. “e83a1c294b5d138339149fcb” and “7d401b038d6fdb08f4f8197d”). It also enabled us to speed up our submission pipeline, since we would only have to generate ~15% of the test predictions.


## Waypoint generation
There is a lot more structure to the waypoints than merely filling the empty space in corridors. We built our solution to attempt to fill plausible waypoint locations without adding too much noise, especially in sensitive areas close to train waypoints.

Our approach consists roughly of: 
- Get a clean map of the corridors from the floor map GeoJSON
- Identify waypoints along walls based on the distance to the nearest wall.
- Gather statistics about nearest neighbor euclidean distance, distance to the wall and distance between points along the wall (project wall points to the wall line)
- Fill in open spots along the wall line using the distance stats and linear referencing. Corners get special consideration.
- Fill inner points aligned with known and generated wall points.
- The room layout may give overlapping generated waypoints. Resolve these to the centroid of small local clusters based on global distance stats or medium local cluster stats.
- Apply a hierarchy of filtering too close points: Known waypoints > corner points > wall points > inner points

Because it is hard to trust CV in this challenge, the parameters were tuned by hand and we mainly used our eyes to assess progress.

Example of our generated waypoints:

[](https://www.kaggle.com/c/indoor-location-navigation/leaderboard)


## Optimization
The discrete optimization is at the heart of our solution. It was obvious that search for the best trajectory would have to scale linearly as a function of the trajectory length, so exhaustive search was out of the question. Beam Search to the rescue! The optimization works as follows:


A) Start by considering up to 2000 most likely initial waypoints, and compute a penalty for the starting point (only WiFi in our final submission)
B) For the trajectory of length L so far, consider 100 likely candidates for the next waypoint
C) Compute a penalty for the most recent segment, for the 2000\*100 considered options
D) Order the 2000\*100 trajectory candidates of length L+1 by their total penalty
E) Drop the candidates that don’t belong to the top 2000 and go back to B until the trajectory is completed

Our prediction is then simply the trajectory with the lowest overall penalty. During validation, we always keep the best trajectory around, so we can understand where the optimization drops the ball. After weeks of tuning the optimization, we are now confident that we will almost always select the trajectory with the lowest optimization error.
Step B discards next step waypoints that are not in the half plane of the direction of the sensor prediction. We also prefer next step waypoints that are at the approximate predicted direction and distance from the previous waypoint.

In our final submissions, we consider 7 types of penalties:
1. **WiFi**: prefer waypoints where the inference WiFi signal is close to the 20 nearest train neighbors of that waypoint location.
We also generate a small boost for the cosine similarity between the vector of the segment, and the vector of the WiFi best guess. This can be interpreted as: does the WiFi movement agree with the proposed segment direction.
2. **Relative movement angle**: Based on the angle between the last two segments.
3. **Relative movement coordinate**: Based on the independent X and Y differences between the predicted relative movement and the proposed segment. 
4. **Pairwise integrated relative movement coordinate**. We also penalize predictions that are not consistent at the trajectory level. Every (L+1)th waypoint is assessed for compatibility with waypoints 1 through L by integrating the predicted relative movements, and comparing that integrated prediction with the vector from each waypoint to waypoint L+1.
5. **Distance based**: Linearly increasing penalty as you move more or less far between waypoints.
6. **Time leak**: Apply a fixed penalty for not agreeing with the edge points of neighboring trajectories, when those trajectories seem to be at the same location and are close in time. Additionally, apply a linearly increasing penalty for moving further away from reliable edge points.
7. **Off grid penalty**: In order to bias the optimization towards known grid points, we apply a penalty which increases as a function of the nearest known grid point. We also apply an additional penalty for selecting additional grid points in a region of dense grid points.

On top of that, we disallow selecting the same waypoint in two subsequent steps. We also adjust the weight of the sensor penalties based on the uncertainty of those predictions (achieved through the time and device leak).

The hyperparameters were tuned with Bayesian optimization. We spent a lot of time looking at our prime misclassifications and estimate that more than half of the error we make is due to inconsistencies in the data. 

Below you see a snapshot of the outcome of the optimization for a test trajectory, together with the most relevant predictions that the optimization builds on.

[](https://www.kaggle.com/c/indoor-location-navigation/leaderboard)


## Ensembling
During the last two days, we had a hard time agreeing on what additional waypoint grid to choose. If you add too many additional waypoints, the optimization can sometimes pick shifted trajectories. However, if you don’t add enough waypoints, the predictions can be drastically wrong, because of the discrete nature of the optimization.

In the end, we realized that we didn’t have to choose a single grid! Our final submissions generate predictions with 3 different grids:
1. Only additional wall waypoints
2. Additional wall waypoints + sparse inner waypoints
3. Dense additional wall waypoints + dense inner waypoints

We select the prediction with the lowest optimization penalty, where our submissions vary in the priority corrections. Ensembling enabled us to mostly stick with the simple grid, except when it resulted in a significant drop of the optimization penalty. Both final submissions boosted our score by about 10cm, relative to only using the sparse grid. 


## Final thoughts
We are happy to share all of our code in [this public repository](https://github.com/ttvand/Indoor-Location-Navigation-Public). The repository contains all our competition code, both the used and unused bits of our final solution. We also added a main script which should generate our approximate final submissions.
