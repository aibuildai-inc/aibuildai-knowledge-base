# #4 solution: Feature engineering made the difference

Competition: playground-series-s3e11
Rank: #4
Source: https://www.kaggle.com/c/playground-series-s3e11/discussion/399489

Thanks Kaggle for hosting this Playground series. These synthetic datasets allow many different experiments to be performed quickly.

My solution is basically a blending of a total of 5 LightGBM (2x), XGBoost (2x) and Catboost (1x) models.

**Data**
- I added the original data to my cv-process. I used the original data for the training process of each fold, but validated exclusively on the synthetic competition data:

- I therefore mark the original data with fold -1, so that it is not used as a validation dataset in any iteration
- I transformed the target variable with np.log(cost) and used rmse as objective
- I used a subset of the original variables: 'store_sqft', 'florist', 'salad_bar', 'prepared_food', 'coffee_bar', 'video_store', 'total_children', 'avg_cars_at home(approx).1', 'num_children_at_home'

**Feature Engineering**
- I spent a lot of time on feature engineering. The most important feature was an overall score for the store-specific attributes:
- `store_features= ['coffee_bar', 'video_store', 'salad_bar', 'prepared_food', 'florist']`
- `df['store_score'] = df[stores_features].sum(axis=1)`
- I also calculated the ratio in relation to the store size:
- `df['store_score_ratio'] = df['store_sqft'] / df['store_score']`
- With these two features, I was able to train a single-lightgbm model, which alone had a public score of 0.2926 and a private score of 0.29326 (private leaderboard range from 9 to 37). The cv score was also much better:

- It was important to pass the new features to LGBM as categorical variables
- For XGBoost and Catboost, I was not able to get this significant improvement. I formed another feature for this and did not include the ratio:
- `(df['florist']*3) + (df['food_proxy']*2) + df['coffee_bar'] + df['video_store']`
- df['food_proxy'] is the sum of prepared food and salad bar and then captured at 1 and I have passed the feature as a numeric feature.
- The feature engineering was the biggest boost for me compared to the public solutions

**Ensembling**
- Simple weighted blend optimized with Optuna

**What didn't work**
- Blending with multiple models and different feature subsets
- Stacking was worse for me with each variant
- I tried to clean up the inconsistent values of the store features based on the original data, however the results got worse that way
- Models other than LGBM, XGB, and Catboost were significantly worse and didn't add value when blending either

Thank you all for the interesting competition. I am looking forward to the next parts of the playground series.
