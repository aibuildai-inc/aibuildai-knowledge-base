# 1st place. That was unexpected...

Competition: playground-series-s3e3
Rank: #1
Source: https://www.kaggle.com/c/playground-series-s3e3/discussion/380920

I hadn't even selected any submissions because I wasn't expecting to do well as I watched my public LB score slip. 😄

Many thanks to [Khawaja Abaid](https://www.kaggle.com/khawajaabaidullah), whose notebook [Starting Strong - XGBoost + LightGBM + CatBoost](https://www.kaggle.com/code/khawajaabaidullah/starting-strong-xgboost-lightgbm-catboost) was the basis of [my own](https://www.kaggle.com/code/bcruise/starting-strong-xgboost-lightgbm-catboost?scriptVersionId=116642571). Please go upvote Khawaja's notebook if you haven't already. My only big change was to add some feature engineering before training the same models. I had discussed it previously in [Adding Risk Factors](https://www.kaggle.com/competitions/playground-series-s3e3/discussion/378994), but here's the final FE code from the winning version:

```python
df['MonthlyIncome/Age'] = df['MonthlyIncome'] / df['Age']

df["Age_risk"] = (df["Age"] < 34).astype(int)
df["HourlyRate_risk"] = (df["HourlyRate"] < 60).astype(int)
df["Distance_risk"] = (df["DistanceFromHome"] >= 20).astype(int)
df["YearsAtCo_risk"] = (df["YearsAtCompany"] < 4).astype(int)

df['NumCompaniesWorked'] = df['NumCompaniesWorked'].replace(0, 1)
df['AverageTenure'] = df["TotalWorkingYears"] / df["NumCompaniesWorked"]
# df['YearsAboveAvgTenure'] = df['YearsAtCompany'] - df['AverageTenure']

df['JobHopper'] = ((df["NumCompaniesWorked"] > 2) & (df["AverageTenure"] < 2.0)).astype(int)

df["AttritionRisk"] = df["Age_risk"] + df["HourlyRate_risk"] + df["Distance_risk"] + df["YearsAtCo_risk"] + df['JobHopper']
```
