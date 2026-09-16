# 37th Place Solution - Feature Engineering + Ensembling

Competition: predict-energy-behavior-of-prosumers
Rank: #37
Source: https://www.kaggle.com/c/predict-energy-behavior-of-prosumers/discussion/499358

First, we would like to thank the organizers and Kaggle for hosting such an interesting challenge. Here is a brief overview of our solution with @sercanyesiloz that stayed robust over the 3 months. We finished the public part around 150th place and jumped to 45th and 30th in the first two updates respectively. And 37th place became our final position with the last update.

We started this competition a little late so we used @vitalykudelya's [notebook](https://www.kaggle.com/code/vitalykudelya/enefit-object-oriented-gbdt) as a starting point. 

We used the last 3 months of training data as our CV setup. For the submission, we used the whole data available. Here is the overview of the new features that helped the most both in CV and public LB:

```python
["week", "quarter"]
is_morning=pl.col("hour").is_in([6, 7, 8, 9, 10, 11]),
is_midday=pl.col("hour").is_in([12, 13, 14, 15]),
is_afternoon=pl.col("hour").is_in([16, 17, 18, 19]),
is_evening=pl.col("hour").is_in([20, 21, 22, 23]),
is_night=pl.col("hour").is_in([0, 1, 2, 3, 4, 5]),

pl.when(pl.col("datetime").dt.month_start() == pl.col("datetime")).then(1).otherwise(0).alias("is_month_start"),
pl.when(pl.col("datetime").dt.month_end() == pl.col("datetime")).then(1).otherwise(0).alias("is_month_end"),

["sin(month)", "cos(month)"]

df["wind_magnitude"] = np.sqrt((df["10_metre_u_wind_component"] ** 2) + (df["10_metre_v_wind_component"] ** 2))

df["shortwave_radiation/surface_pressure"] = df["shortwave_radiation"] / df["surface_pressure"]

df["production_target"] = df["installed_capacity"] * df["surface_solar_radiation_downwards"] / (df["temperature"] + 273.15)
```

**wind_magnitude** feature is derived from this blog post: http://colaweb.gmu.edu/dev/clim301/lectures/wind/wind-uv

The second important aspect of our solution was the ensembling. We combined the LGBM model with a neural net. Even though the score of the neural net was lower than the LGBM itself (3-4% on public LB), it gave us a significant boost in the final score. Here the second one is the ensemble with a neural net: 


The first one is the LGBM model with refitting every 31 days.

We were not 100% sure about our refit strategy, so we kept the ensemble submission with no refitting (it was always stronger). Given our experiments gave a 1% boost overall with a refit, It would probably end up in the top 25 including that.

We used `target` and `target_diff=target-target_48h` as our prediction objective. `target / installed capacity` objective did not work in our setup as some other solutions claim to be effective.
