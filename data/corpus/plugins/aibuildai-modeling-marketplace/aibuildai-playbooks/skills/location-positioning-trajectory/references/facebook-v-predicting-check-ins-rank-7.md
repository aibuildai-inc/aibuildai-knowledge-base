# Solution Sharing

Competition: facebook-v-predicting-check-ins
Rank: #7
Source: https://www.kaggle.com/c/facebook-v-predicting-check-ins/discussion/22078#126195

jturkewitz Thanks for sharing the "peek into the future" idea. Clever thinking!

My best submission (place 7) is an ensemble of three models.

**1. Regression Model**

This model performs logistic regression using XGBoost to determine the likelihood of a pairing of a checkin and a place. For each observation in the training set, 50 observations are derived. The actual place serves as the positive observation, and the 49 closest places serve as negative observations. Training observations thus grow by a factor of 50, and only 20% of the original training data was used to build this model. Gains in precission became diminishing after that.

The features of this model are based on a set of pre-computed per-place data, such as as mean and standard deviation of location and accuracy, visitor frequencies, and neighborhood sparsity. Outlier removal using Turkey's test was used for some of these statistics.

In particular, the model features include:

- Basic: x, y, accuracy
- Location: L1 and L2 distances with/without normalization, uni- and multi-variate normal PDFs of location and accuracy (all calculated on the observation data with regard to the pre-calculated per-place data of the place under consideration)
- Frequencies: time of final visit, mean visitors on the particular day/weekday/hours, using a shifted weekday concept (+ 12 h) for places peaking across midnight
- Neighborhood: L1 and L2 sparsity of the place as determined by mean distance to the next 25 neighbors
- Comparisons: min/max/delta on location and various visitor frequencies of the place and its nearest neighbors.

Individual MAP@3 on the public leaderboard: 0.58452

While this is not the best performing individual model, a strong point of this model is its high degree of generalization. Adding predictive support for a new place only requires calculating the per-place statistics for that place. This is different from the other two models which need to be partially re-trained whenever the set of places changes.


**2. Classification Model**

This is a grid-based classification model using XGBoost to directly classify an observation to a place. 25 X cuts, and 100 Y cuts are used. Instead of heavy border augmentation, four instances of the model are built, each one shifted by a quarter of a grid.

Features include:

- Basic: x, y, accuracy, time
- Periodic time: day time (% 1440), weekday time (% 7 * 1440)
- Time factors: weekday, shifted weekday, hour of day, hour of week, month of year

Individual MAP@3 on the public leaderboard: 0.59420


**3. Nearest Neighbor Model**

This is pretty much the awesome and extremely efficient model developed Sandro, Alex, Michael Hartmann, and many others. As with the classification model, four instances of the model are built, each one shifted by a quarter of a grid.


**4. Ensemble**

For each model instance and test observation, the top 6 places and related probabilities are stored.

Ensembling then consists of two phases. In the first phase, the model instances are aggregated per main model to derive the average probability of a place by that model; in the second phase, a weighted average of the probabilities from the three main models is calculated. The submission then consists of the three places with the highest average probability.


Thanks to Facebook for coming up with an awesome problem, based on a simple data set consisting of only 5 attributes, but providing significant depth. And thanks to the Kaggle community! The discussions on the forums and the activity in the scripts have really been a highlight of this competition.
