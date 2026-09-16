# Giba RAPIDS SVR Solution

Competition: godaddy-microbusiness-density-forecasting
Rank: #32
Source: https://www.kaggle.com/c/godaddy-microbusiness-density-forecasting/discussion/395011

For my first solution I used a global linear multiplier were I pick the coefficient optimizing SMAPE for each split month and forecast span separately.

The second solution is based in SVR. You can check my notebook here: [Super Fast RAPIDS SVR](https://www.kaggle.com/code/titericz/super-fast-rapids-svr/notebook)

Once `microbusiness_density` can be calculated using `active` and `county population`, I used `active` as the target labels in my models.
For validation I used last 12 months to calculate the SMAPE for each forecast range. To ease development I tracked only the gain each algorithm gives compared with last value benchmark. For example, predicting 3 months ahead with last value gives SMAPE 2.717, if my algorithm scores 2.617, then the gain will be 0.10 over the baseline.

The SVR gain for each forecast period is:
| Forecast | SMAPE Gain(12 months avg) |
| --- | --- |
| 1 | 0.0171 |
| 2 | 0.0611 |
| 3 | 0.1555 |
| 4 | 0.2621 |
| 5 | 0.3524 |
| 6 | 0.4750 |


As you can see the gain is not much. But all other algorithms I tried (including GBDTs) scored worse than SVR in my approach.

# Other than that I wish good luck to everyone that dedicated time to this competition! 😉

obs. 
- notebook V1 using sklearn SVR in 4607s.
- notebook V2 using RAPIDS SVR 136s. (34x faster 💪)
