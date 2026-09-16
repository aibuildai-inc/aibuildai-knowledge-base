# 13th place solution - pmts_year_1139T postprocess

Competition: home-credit-credit-risk-model-stability
Rank: #13
Source: https://www.kaggle.com/c/home-credit-credit-risk-model-stability/discussion/508113

First of all, thanks to Kaggle and the host of the competition. To be honest, we don't fully understand what was effective, let me briefly describe our solution.

Thanks @kentookumura @pegasus27 for the collaboration.

## Context

Business context: [https://www.kaggle.com/competitions/home-credit-credit-risk-model-stability/overview](https://www.kaggle.com/competitions/home-credit-credit-risk-model-stability/overview)

Data context: [https://www.kaggle.com/competitions/home-credit-credit-risk-model-stability/data](https://www.kaggle.com/competitions/home-credit-credit-risk-model-stability/data)

## **Overview of the approach**

In feature engineering, we created 772 handcrafted features using tables other than credit_bureau_b1, other_1, deposit_1, debitcard_1. We performed manual and correlation-based feature selection and reduced the number of features to 411. 

In modeling, we built 10 models using LightGBM, XGBoost, CatBoost, and HistGradientBoostingClassifier. I stacked these outputs with a RidgeClassifier and applied probability calibration to create predicted values. Then, we created the final predicted values using random seed averaging.

In postprocess, we added a simple process to take the maximum value of pmts_year_1139T in the credit_bureau_a_2 table and apply a negative correction to the score for each year.

## **Details of the submission**

### Feature Engineering

The feature engineering was mainly done by @kentookumura. Initially, we decided not to use the credit_bureau_b_2, other_1, deposit_1, debitcard_1 tables which had a high rate of missing case_id. We created handcrafted features based on the results of EDA and the solutions from past competitions. We reduced the number of features from 772 to 411 through manual and correlation-based feature selection. Below, we will describe the points that we believe were effective.

**Era**

```python
df_base = df_base.with_columns(
		((pl.col("first_birth_259D") / 10).floor() * 10).alias("era").cast(pl.Int32),
)
```

**Age at Start of Employment**

```python
df = df.with_columns(
        ((pl.col("empl_employedfrom_271D") - pl.col("birth_259D")).dt.total_days() // 365).cast(pl.Int32).alias("agestartofemploymentA"), 
)
```

**Employment Period**

```python
df_base = df_base.with_columns(
        (pl.col("first_birth_259D") - pl.col("first_agestartofemploymentA")).alias("durationofemploymentA"),
)
```

**Date processing other than suffix D**

```python
def handle_dates(df):
    for colin df.columns:
        if col[-1]in ("D",):
            df = df.with_columns(pl.col(col) - pl.col("date_decision"))
            df = df.with_columns(pl.col(col).dt.total_days())
            df = df.with_columns(pl.col(col).cast(pl.Float32))

				elif "year" in col:
            df = df.with_columns(pl.col(col) - pl.col("date_decision").dt.year())
            df = df.with_columns(pl.col(col).cast(pl.Int32))
```

**Merge tax_registry tables**

From some case_id with multiple provider information, we inferred the correspondence of each table column and made it into one table.

**Aggregation of String type (mode and n_unique)**
We used the process `pl.col(col).drop_nans().drop_nulls().mode().sort().first()` for reproducibility in polars.

**Removal of features that fluctuate greatly during the training data period**

We manually checked and removed features that fluctuate greatly with each WEEK_NUM.

### Modeling

@uplus26e7 mainly handled the modeling. We used StratifiedGroupKFold(k=5) based on WEEK_NUM for CV. We tried multiple GBDT models that do not require scaling of features or missing value completion, as these did not go well. In order to create diverse models, we created 10 models with multiple parameters and stacked them with RidgeClassifier. Finally, we corrected the predicted values using Scikit-Learn's CalibratedClassifierCV.

The above model performed random seed averaging (5 seeds) and used it for the final inference.

| Model | Local CV AUC (average 5 seeds) | Main Parameters |
| --- | --- | --- |
| XGBoost | 0.8569388045 |  |
| CatBoost | 0.8543810988 |  |
| LightGBM | 0.8576141787 | boosting=”gbdt”, extra_tree=True |
| LightGBM | 0.8569003931 | boosting=”gbdt” |
| LightGBM | 0.8068982316 | boosting=”rf” |
| LightGBM | 0.7993733596 | boosting=”rf”, extra_tree=True |
| LightGBM | 0.8546620954 | boosting=”dart” |
| LightGBM | 0.8517172791 | boosting=”dart”, extra_tree=True |
| HistGradientBoostingClassifier | 0.8496922895 |  |
| LightGBM | 0.8574351195 | boosting=”gbdt”, extra_tree=True, data_sample_strategy=”goss” |
| CalibratedClassifierCV (RidgeClassifier) | 0.859322 |  |

### Postprocess

@kentookumura's thorough EDA and experiments revealed that the 'pmts_year_1139T' in the 'credit_bureau_a_2' table is likely the most recent 'date_decision' year. It was also observed that no date column transformations were added in data changes. Based on these findings, we implemented post-processing to decrease the predicted value based on the maximum 'pmts_year_1139T' value.

```python
submission = pd.read_csv("submission.csv")
pmts_year = ... # max pmts_year_1139T group by case_id
    submission.loc[pmts_year == 2020, "score"] = (submission.loc[pmts_year == 2020, "score"] - 0.07).clip(0)
    submission.loc[pmts_year == 2021, "score"] = (submission.loc[pmts_year == 2021, "score"] - 0.06).clip(0)
    submission.loc[pmts_year == 2022, "score"] = (submission.loc[pmts_year==2022, "score"] - 0.02).clip(0)
    submission.to_csv("submission.csv", index=False)
```
