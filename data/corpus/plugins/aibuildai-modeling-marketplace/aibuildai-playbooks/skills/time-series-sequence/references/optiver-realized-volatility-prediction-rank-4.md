# Tentative 3rd Place Solution (6th in Public) - life is volatile

Competition: optiver-realized-volatility-prediction
Rank: #4
Source: https://www.kaggle.com/c/optiver-realized-volatility-prediction/discussion/278676

Before I start, I would like to thank kaggle and Optiver for hosting this phenomenal competition. And, of course, my great teammate, @tomotomo5, who drafted this post.
 
We have ended off 6th place in the Public Leaderboard and are 3rd place currently (as of October 15th) in the Private Leaderboard.

In this discussion, we would like to introduce the core ideas we have implemented that boosted our score. Some of them are not yet discussed much in the community.  Model-wise, our final model is a simple stacking model of LightGBM x ANN (less noteworthy).

## time_id Nearest Neighbor
The largest gamechanger of our solution was features based on time_id nearest neighbors, which was a similar idea of feature creation @nyanp has explained in his solution below.
 https://www.kaggle.com/c/optiver-realized-volatility-prediction/discussion/274970
 
I confess that our philosophy was not as sophisticated as @nyanp's. We just tried to find the time ids that have similar market conditions. In fact, we were not quite aware that we could recover time-id order, although we ended up creating quite similar features to @nyanp's.
 
Differences are,
+ ~~No usage of dimension reduction (TSNE in @nyanps solution) when calculating the nearest neighbors~~
+ Adding the distance ratio of the 1st nearest time_id and k-th nearest time_id
- No time-series cross-validation as we did not come up with the idea of time-series reverse engineering:(
 
## Target Transformation
 Target Transformation was a crucial part of our model that increased the robustness. Stock volatility could behave as a non-stationary series, meaning that the level of volatility might be unstable over time but the change of volatility is less likely so.
 
With this in mind, we decided to change the task of the competition from directly predicting the level of realized volatility to predicting the ratio of the target to the realized volatility of 0~600 seconds.

 Specifically, our transformed target is formulated as follows.
`(transformed_target) = (target) / (realized volatility of 0~600 seconds)`
## 300 seconds Model
 Since a large time interval between the training dataset and the final test dataset set was expected, we wanted to give our model as much information on the test dataset as possible.

 Therefore, we (1) concatenated the train + test set, (2) sliced the 600 seconds in half, (3) used the first half (0-300 seconds) to create features (4) to predict the realized volatility in the latter 300-600 seconds. We then (5) created features based on 300-600 seconds of the dataset, and (6) predicted the realized volatility of 600-900 seconds. 

 The predicted results were used as features of our main model. 
 The best part of this idea was that we were able to train our model based on test data.

## Macro Estimation
 We believed that individual stock volatility depends largely on the macro environment.
 Therefore, we trained a model to predict the average volatility of all stock_ids in each time_id.

 The results were used as features of our main model.



**Hope the best that the private leaderboard thereafter is not that volatile even if *life is volatile*.**
**Thank you so much for reading, HAPPY KAGGLING!**
