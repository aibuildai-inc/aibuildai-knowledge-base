# 2nd Place Solution

Competition: mlb-player-digital-engagement-forecasting
Rank: #2
Source: https://www.kaggle.com/c/mlb-player-digital-engagement-forecasting/discussion/274661

Our final model was a blend of 6 GBM models (5 Lightgbm, 1 Xgboost) all trained on the same features. We validated on the last month of data and then retrained the model on the entire dataset.

- LightGBM - objective=”regression_l1”
- LightGBM - objective=”regression_l2” - targets scaled by double square root
- LightGBM - objective=”regression_l1” - boost_from_average=True
- LightGBM - objective=”regression_l1”, 'max_depth': -1
- LightGBM DART - object=’regression_l1’, boosting=’dart’
- Xgboost - targets scaled by double square root

Our final feature set had over 1000 features that pretty fell under 3 categories:

- Target aggregates - All-time and rolling 12-month historical mean/variance of target for each player.
- Last 20 days/games of various stats for each player. For example, `strikeouts_1_day_ago`, `strikeouts_2_days_ago`, ..., `strikeouts_20_days_ago` for each player. 
- Features based on our baseball knowledge/what gets people to tweet. For example, a walk-off hit/home run will often trigger some engagement. Other features were no-hitters, win probability added by a player (a proxy for if they played a good game or were involved in "high leverage" plays), ranking in the home run race, was a player ejected, ERA and ERA ranking, and more.


Some of our most important features were:
- numberOfFollowers - most recent value for a players number of Twitter followers
- numberOfFollower_delta - the change in Twitter followers between the most recent 2 months
- monthday - integer value representing current month and day (mmdd) e.g. June 4th would have a value of `604`.
- {target}_p_var - Historical variance of {target} for player
- {target}_p_gameday_mean - Historical mean of {target} for player on gamedays
- roll12_{target}_p_mean - {target} mean for player for the previous 12 months
- roll12_{target}_p_var - {target} variance for player for the previous 12 months
- wpa_daily_max - League-wide daily maximum value of Win Probability Added. An approximate (exact base-out states are not available, so calculation is approximate) WPA value is calculated for each player. 
- homeRuns_rank - Players’ home run ranking 
- walk_off_league - Was there a walk off hit in the league that day?
- days_since_last_start - number of days since a player last pitched. Useful for the model to learn if a player is likely to pitch on the day engagement is measured.

Submission Notebook: https://www.kaggle.com/brandenkmurray/mlb-predict-final
