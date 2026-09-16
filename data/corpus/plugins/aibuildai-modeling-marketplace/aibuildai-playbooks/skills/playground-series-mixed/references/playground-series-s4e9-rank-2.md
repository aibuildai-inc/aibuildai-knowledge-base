# #2 Position | Just FE and AutoML

Competition: playground-series-s4e9
Rank: #2
Source: https://www.kaggle.com/c/playground-series-s4e9/discussion/537349

Hello everyone,

I'd like to share my approach to this competition, in case anyone is interested.

For feature engineering, I took the following steps:

- Simplified transmission values by consolidating "Automatic" and "Manual" into "A/T" and "M/T", respectively
Extracted luxury brands from the data
- Derived features from the engine column, including horsepower, cylinders, and combinations of these
- Created feature crosses, such as int_ext_col, brand_model, brand_int_col, brand_ext_col, and brand_mileage
- Identified and marked infrequent categories for each feature as noise, based on quantiles
- Used car age and mileage to create a mileage_per_year feature
- Handled missing values, of course

For modeling, I experimented with CatBoost Regressor and LGBM Regressor using Optuna, but my best results came from a Weighted Ensemble fitted with AutoGluon. I was able to further improve my score by incorporating additional data.

I have to say, I'm still in shock from the final leaderboard shake-up! I didn't expect to end up in second place, especially considering I had very limited time to try out different approaches. I'm thrilled and grateful for the outcome.

I also want to extend a big thank you to @roberthatch and @cdeotte for the insightful discussions and contributions throughout the competition. It was a great experience, and I'm glad I got to be a part of it.

Thanks again, and congratulations to all participants! ✌️
