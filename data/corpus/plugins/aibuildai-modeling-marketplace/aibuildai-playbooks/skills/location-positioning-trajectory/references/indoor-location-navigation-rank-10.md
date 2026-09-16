# 10th Place Rapids cuML Solution Quick Writeup

Competition: indoor-location-navigation
Rank: #10
Source: https://www.kaggle.com/c/indoor-location-navigation/discussion/239905

This is a long and hard competition! I'd like to congratulate all the teams who stick to it till the end. I am looking forward to the sharing of winning solutions.

My main goal of this competition is to experiment with [Rapids.ai](https://rapids.ai/start.html) tools to accelerate the pre-processing and post-processing of deep learning models. 

Pre-processing:
- use `dask` to process raw data, convert them to dataframes and save them as `parquets`.
- use `dask-cudf` to engineer features from many small `parquets`. 
- use `cuml LabelEncoder, TargetEncoder, Nearest Neighbor` and `Xgboost` to create simple models and explore the dataset.

Model:
I built two RNNs using PyTorch Lightning:
- use wifi features to predict the waypoints directly. 
- use IMU features to predict the shift of waypoints, `delta x & y`. similar to [Olaf's approach](https://www.kaggle.com/c/indoor-location-navigation/discussion/239884)

Post-processing:
The approach is to interleave [cost minimization](https://www.kaggle.com/saitodevel01/indoor-post-processing-by-cost-minimization), [snap to corridor] (https://www.kaggle.com/rafaelcartenet/scaled-floors-geojsons-new-dataset) and [snap to grid](https://www.kaggle.com/robikscube/indoor-navigation-snap-to-grid-post-processing) and run them iteratively. Hyperparameters such as `alpha` in `cost minimization`, `threshold` in `snap grid` are tuned for each iteration. I rewrote these functions with `cupy` and `cudf` and they were very fast. My best submission is done with 30 iterations and 90 post-processing functions in total, which only took less than 10 minutes. I didn't do the exact measurements but that could be 100x faster than the original implementation. The speedup is very important for searching for the best hyperparameters. The post-processing improves the score by 1.1~1.2, which is quite significant.

Other notes:
I didn't use the start/end points leak and the hand-labeled waypoints. I spent a lot of time creating an end-to-end neural network that incorporates `cost minimization` and `snap to grids` in the training process but it didn't go anywhere in the end. I'm looking forward to the approach of the 1st place team who successfully built an end-to-end model.

Unfortunately, I am way behind in my other workloads so I apologize that my solution and source code will be shared later when I find the time. I'll update this thread when it's done.
