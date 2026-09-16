# My Forecasting Strategy

Competition: godaddy-microbusiness-density-forecasting
Rank: #4
Source: https://www.kaggle.com/c/godaddy-microbusiness-density-forecasting/discussion/394821

# Trend
My cross-validation didn't show that complicated gradient boostings showed significantly better results than simple approaches. Also for some months  gradient boostings performed worse than the Last Value baseline and simple approaches showed more stable performance. 

The following formula shows how I estimated trend for cfips.
`active_quantile_group_21_last_6m_trend` means the best multiplication constant (clipped from 1 to 1.006) for the last value of `microbusiness_density` for the next month on a previous 6 month for a group of cfips (total 21 groups by quantile of active). The same logic for `state` and concrete `cfips` group.
`forward_2` means the best multiplication constant for the next 2 month (clipped from 1 to 1.012).

```python
df_features_data['trend'] = (
    df_features_data['active_quantile_group_21_last_6m_trend'] * 0.16
    + df_features_data['active_quantile_group_21_last_3m_trend'] * 0.16
    + df_features_data['state_last_6m_trend'] * 0.14 + df_features_data['state_last_3m_trend'] * 0.14
    + df_features_data['cfips_last_6m_trend'] * 0.20 + df_features_data['cfips_last_3m_trend'] * 0.20
) * 0.5 + ((
    df_features_data['active_quantile_group_21_last_6m_trend_forward_2'] * 0.4
    + df_features_data['state_last_6m_trend_forward_2'] * 0.4
    + df_features_data['cfips_last_6m_trend_forward_2'] * 0.2
)**0.5) * 0.5
```

# Using the best public submission as a starting point for predictions
Public Leaderboard data wasn't published and for time-series data it should be beneficial to use the best public submission.

# Select 2 Submissions. Positive and Negative
It's a good idea to cover different scenarios selecting 2 submissions. I call it Positive submission (we under-forecasting trend) and negative (we over-forecasting trend). 
For example for Positive submission `* (trend + 0.0005)**(month_number)`

# May Seasonality
I just used forecast for April `* (1 - 0.0025)` for Positive submission, because May showed a dropdown of values the last two years.

# Cfips with active < 150
My cross-validation showed that using the last value baseline for such cfips (as many public baselines do) is good only for one month forward forecast, for March, April, May i used some multiplication constants.
The following plot shows that for active < 75 using the last value baseline for 3 month forecast is the best, for 75-150 it's better to use trend.
 .png?generation=1678841088819412&alt=media)

# Hope that random helps 🤞
