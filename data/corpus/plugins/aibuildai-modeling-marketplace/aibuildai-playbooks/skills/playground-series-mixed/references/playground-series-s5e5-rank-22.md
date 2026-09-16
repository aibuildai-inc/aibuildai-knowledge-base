# 22nd place solution: I asked ChatGPT for feature transformation ideas

Competition: playground-series-s5e5
Rank: #22
Source: https://www.kaggle.com/c/playground-series-s5e5/discussion/582604

## Feature engineering & transformation

I'm not an expert in workout so I asked ChatGPT for some feature transformations that are functions of the original features:
```
def mifflin_st_jeor(row):
    """Basal Metabolic Rate (kcal / day).  Height in cm, Weight in kg."""
    if row["Sex"] == "male":
        s = 5
    else:  # female
        s = -161
    return 10 * row["Weight"] + 6.25 * row["Height"] - 5 * row["Age"] + s

def boer_lbm(row):
    """Lean Body Mass (kg)."""
    if row["Sex"] == "male":
        return 0.407 * row["Weight"] + 0.267 * row["Height"] - 19.2
    else:
        return 0.252 * row["Weight"] + 0.473 * row["Height"] - 48.3

def body_surface_area(row):
    """Mosteller BSA (m²)."""
    return np.sqrt(row["Height"] * row["Weight"] / 3600)

def body_fat_pct(row):
    bmi = row["BMI"]
    adj = 10.8 if row["Sex"] == "male" else 0
    return 1.2 * bmi + 0.23 * row["Age"] - adj - 5.4

def max_hr(age):
    """Age-based Max Heart Rate."""
    return 220 - age

def vo2_est(hr, age):
    """Very rough HR→VO₂ regression (ml·kg⁻¹·min⁻¹)."""
    return 14.0 + 0.37 * hr - 0.006 * age

train_df["BMI"] = train_df["Weight"] / (train_df["Height"] / 100) ** 2
train_df["Total_Exertion"] = train_df["Duration"] * train_df["Heart_Rate"]
train_df["Heart_Effort"] = train_df["Heart_Rate"] / train_df["Duration"]
train_df["BMR"] = train_df.apply(mifflin_st_jeor, axis=1)
train_df["LBM"] = train_df.apply(boer_lbm, axis=1)
train_df["BSA"] = train_df.apply(body_surface_area, axis=1)
train_df["Body_Fat_Pct"] = train_df.apply(body_fat_pct, axis=1)
train_df["MHR"] = train_df["Age"].apply(max_hr)
train_df["pct_MHR"] = train_df["Heart_Rate"] / train_df["MHR"]
train_df["Training_Load"] = train_df["pct_MHR"] * train_df["Duration"]
train_df["VO2"] = vo2_est(train_df["Heart_Rate"], train_df["Age"])
train_df["Heat_Index"] = (train_df["Body_Temp"] - 37.0) * train_df["Duration"]
```
## Cross-validation
5 folds splitting based on 100 quantile bins of `sklearn.neighbors.LocalOutlierFactor.negative_outlier_factor_`. I'm not sure how much this helps but it does detect those non-physical data points and helped decreasing the variance of RMSE between folds. The plot below shows the data points in the bin number `0` (purple) vs. points outside it (yellow).

[lof_scores]

## Modeling
Combining this with the original features, I tuned the three gradient boosting algorithms XGB, LGBM and CatBoost with Optuna on Kaggle notebook. The two submissions I made were [RidgeCV](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.RidgeCV.html) ensembles. The best one got `0.05873` CV and `0.05851` private LB.
