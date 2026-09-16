# 25th place solution

Competition: m5-forecasting-accuracy
Rank: #25
Source: https://www.kaggle.com/c/m5-forecasting-accuracy/discussion/163564

I learned a lot from this competition. I appreciate excellent discussion and notebook.
My solution may be worthless and I wonder why this works well in private, but I share it.
I predict level 1~10 and 12 values.

[model]
lightgbm
level 12 recursive, objective poisson
level 1~10 day-by-day (target-&gt;sale*price)  objective rmse (level 1~9) tweedie (level 10)

[feature]
simple  shift roll diff, product count , some calender features, price

[validation]
d1830-d1857, d1858-d1885, d1886-d1913, d1914-d1941

I spent time on level1~10 prediction, cv of level 1~10 predictions  were stable.
I tuning level1~10 parameters using StratifiedFold.

I combine prediction like this,
`df[level12]*=df[price]`
`df[level_] = df[level_prediction]*(df[level12]/df.groupby(level_).transform(sum))`
`df[final_prediction] = 0.65*mean(df[level8, level9]) + 0.35*mean(df[level1~7, level10, level12])`


I cloud not get good level 12 wrmsse, but get better score when combining the predictions of level 1~10. I believe level1~10 predictions boost wrmsse score.

|  |  single level 12 | combine level 1~10|
| ---  | --- |
|d1942-d1969|0.73233|0.55895|
|d1914-d1941 | 0.60359 |  0.5635|
| d1886-d1913| 0.56219 |  0.51489|
|d1858-d1885 | 0.67186 |  0.60522|
|d1830-d1857 | 0.71142 |  0.644|

I post clearn [notebook](https://www.kaggle.com/mothermoto/m5-solution/notebook).
