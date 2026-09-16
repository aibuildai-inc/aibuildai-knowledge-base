# Public 10th Place Solution (Private 11th)

Competition: predict-energy-behavior-of-prosumers
Rank: #11
Source: https://www.kaggle.com/c/predict-energy-behavior-of-prosumers/discussion/472537

I expect some shake-up but this solution can probably land between 5th and 50th place if it doesn't throw an error. Here is what I did:

**Stacking**

Trained 2 separate models:
- Weather forecast features only model with target/installed_capacity as target
- Weather history features only model with target/installed_capacity as target

Generated estimated production per capacity features and their interactions for my final model.

**Problem Translation**

- Split the data into 4 and trained models for each (is_business, is_consumption) pair
- Created baseline prediction as max(target_lag2, target_lag4, target_lag7)
- New target becomes (target + 1)/(baseline_pred + 1)
- For each data partition, trained 3 models with different seeds and different sample weights (equal, log1p(baseline_pred) and sqrt(baseline_pred)

**Features**

76 features. Nothing special.

```python
features = ["baseline_pred", "dow", "month", "hour", "capacity_change", 'eic_change',
            "est_per_capacity", "est", "prod_rate", "cons_rate", "cons_rate2", "consumption_type",
            "county", "product_type", "no_holiday", "capacity_lag_2days", "eic_lag_2days"] + weather_forecast_cols
for day in [2, 4, 7]:
    features += [f"target_lag_rate{day}", f"target_sum_lag_rate{day}", f"target_lag_{day}days",
                 f"target_sum_lag_{day}days", f"target_lag_{day}days_norm", f"est_per_capacity_lag{day}_rate"]
for day in [2, 7]:
    features += [f"{col}_lag{day}" for col in weather_hist_cols]
```

**Postprocessing**

For each prediction_unit_id, calculated lag 2, 3, 4 errors and took their mean. 0.2*avg_error is added into the predictions as correction term. This way, I improve the performance for the cases where I constantly underpredict or overpredict.

**Re-training**

Every 8 days, I re-train my XGBoost models on GPU. This is especially useful when there are new prediction unit ids or increase in eics.

**Validation**
I split last 9 months into 3 validation folds. Also used the public LB as the 4th fold. While my 3-fold CV is 35.73, my public LB is 61.32.

**Ensemble**
Despite its relatively worse score, 80/20 ensembling with the public best kernel made my score 59.97. I think it is because my model was diverse and this problem benefits a lot from diverse ensembling. Btw I had an improvement also in my CV score by ensembling with the public best kernel but it was significantly lower.
