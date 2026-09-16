# 7th place solution

Competition: web-traffic-time-series-forecasting
Rank: #7
Source: https://www.kaggle.com/c/web-traffic-time-series-forecasting/discussion/43621

I like that many guys have shared their ideas.  And the discussion here  is very pleasant. Here is the idea.  

The submission  is  weighted average of  several nn models.  Each model have 1 lstm layer and 2 fc layers. The differences are their inputs or objectives.

The input may be raw numbers or be transformed by log1p or be normalized by average. The  objective may be either MAE or SMAPE. Although SMAPE is not convex, but it still works with SGD.

The solution does not count yearly seasonality, which I think is the major cause for the gap away from the lead.

The models were trained only against 8/31 data because of hurricane irma.  The final submissions were made on a laptop.  But I don't think this is critical.

Enjoy!
