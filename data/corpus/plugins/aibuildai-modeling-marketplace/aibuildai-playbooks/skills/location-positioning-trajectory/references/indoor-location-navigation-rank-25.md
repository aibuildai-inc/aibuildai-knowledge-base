# [25th] Public Sub + Shortest Path Search

Competition: indoor-location-navigation
Rank: #25
Source: https://www.kaggle.com/c/indoor-location-navigation/discussion/239893

# Thank you everyone!

My solution is very simple. Everything is available from [here](https://www.kaggle.com/tomooinubushi/25th-public-sub-shortest-path-search).



**1. I use the results of [public best submission](https://www.kaggle.com/ahmedewida/indoorlocation-ensembling).**
I used the public sub, because my NN models were never better than 7.7 in CV. I am looking forward to see other solutions.

**2. Correct floor predictions based on the leakages of [shared wifi records](https://www.kaggle.com/tomooinubushi/retrieving-user-id-from-leaked-wifi-feature) and [device IDs](https://www.kaggle.com/c/indoor-location-navigation/discussion/234543).**

I assumed that the paths with the same ID are in the same floor, which is not always true for train waypoints. Correcting start/end waypoints based on the leakages did not work well when combining with following shortest path search post-processing. This process changes floor predictions of only three paths that are in private test set i.e., this process did not change public LB score.

**3. Postprocess the waypoints based on [Dijkstra's algorithm](https://en.wikipedia.org/wiki/Dijkstra%27s_algorithm).**
I re-defined the task as a [shortest path problem](https://en.wikipedia.org/wiki/Shortest_path_problem) rather than a regression task. I searched the path with minimal [cost](https://www.kaggle.com/saitodevel01/indoor-post-processing-by-cost-minimization) based on Dijkstra's algorithm, in which the nodes are train and augmented waypoints.

Instead of using hand-labeled waypoints, I generated augmented waypoints with following rules.
-   Augmented waypoints have similar X and Y values of train ones (mean of the subset of train waypoints).
-   Augmented waypoints are in hallways.
-   Augmented waypoints are sufficiently distant from train ones and each other.

I used codes and ideas from many many discussions and notebooks. Please notify me if I miss someone. Thank you very much.

* https://www.kaggle.com/kenmatsu4/feature-store-for-indoor-location-navigation
* https://www.kaggle.com/jiweiliu/fix-the-timestamps-of-test-data-using-dask
* https://www.kaggle.com/ahmedewida/indoorlocation-ensembling
* https://www.kaggle.com/c/indoor-location-navigation/discussion/234543
* https://www.kaggle.com/saitodevel01/indoor-post-processing-by-cost-minimization
* https://www.kaggle.com/museas/with-magn-cost-minimization
* https://www.kaggle.com/higepon/visualize-submissions-with-post-processing
* https://www.kaggle.com/robikscube/indoor-nav-visualize-predictions-train-data
* https://www.kaggle.com/nigelhenry/simple-99-accurate-floor-model
