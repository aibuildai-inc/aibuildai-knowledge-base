# 9th Place Solution

Competition: trends-assessment-prediction
Rank: #9
Source: https://www.kaggle.com/c/trends-assessment-prediction/discussion/162939

I joined that competition pretty late so I started with a brute force attack. 

My solution is basically a big 3 level stack ensemble of tabular models. 
My final prediction have 24 baseline models.

- 6x Ridge
- 1x Lasso
- 3x RAPIDS SVR
- 2x XGboost
- 2x KNN
- 9x MLP
- 1x 2D CNN

Each model is trained using different datasets including the original features and additional engineered features extracted from 3D brain scans.


For the second level stacked predictions I used 4 models: 
- BaggingRegressor( base_estimator=Ridge )
- KNeighborsRegressor
- NuSVR
- RandomForestRegressor

My final prediction is a weighted average of those 4 model. My final local CV is 0.1559 and LB: 0.15724 (Public)/ 0.15742(Private). I believe the gap between local and LB scores are due some site2 differences.

Each fMRI scan have 4 dimensions (53, 52, 63, 53) and it makes things hard when the topic is extracting features. Basically the first axis represents 53 different scans (not time related) so the dimension of each brain scan is only (52, 63, 53).

To extract features from fMRI data I used many approaches, but the ones that performed better were when I calculated metrics comparing images from two different scans. So, most of my features are 2-way interactions of all 53 scans for each patient. That way I created thousands of features to describe each fMRI. 

To make the MLP works better I had to use BatchNorm in all layers and Dropout of around 0.7~0.8 to generalizes well. That was a key point in my models since most of the features have a very small signal or are just random noise.

My 2D CNN was trained using projections of each 3D image in the 3 axis: x, y, z. So each fMRI scan is reduced to a 2D representation. That model scored 0.17x locally.

- Autoencoders embeddings didn't worked for me.
- 3D CNNs scored very bad to me and I regret for not using it in my stacking since top solutions used it.
- Post processings site2 worked just a little bit for Public LB.
- Feature Selection overfited trainset, so I didn't used it.
- I'm surprised that using simple KFold for CV worked pretty well for stacking in this small dataset (5877 rows)
- RAPIDS t-SNE features didn't worked.
