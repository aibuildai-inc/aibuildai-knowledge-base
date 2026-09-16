# 6th Place: Select Feature Combinations based on RMSE Scores

Competition: playground-series-s5e4
Rank: #6
Source: https://www.kaggle.com/c/playground-series-s5e4/discussion/575783

Repo: https://github.com/masaishi/kaggle-s5e4-6th-solution


**Update:** I published the training code, and it can achieve an 11.63 private score with a single LightGBM model using 5-fold cross-validation.
Code [Public 11.70 | Private 11.63 | Single LGB Fold 5](https://www.kaggle.com/code/masaishi/public-11-70-private-11-63-single-lgb-fold-5?scriptVersionId=237136503)



## TL;DR
- Identified data leaks(?) and applied targeted corrections
- Original Data as New Rows
- Select feature combinations based on RMSE scores

## Data Leak 1: More than 2 decimal digits in Episode_Length_minutes

I verified the data leak originally shared by AngelosMar in [this discussion post](https://www.kaggle.com/competitions/playground-series-s5e4/discussion/574925#3187345). My own verification EDA can be found here: [Decimal Digits Analysis EDA](https://www.kaggle.com/code/masaishi/decimal-digits-analysis-eda?scriptVersionId=236025089)

The analysis revealed that data points with more than 2 decimal digits in Episode_Length_minutes were strongly correlated with the actual Listening_Time_minutes. After ensemble prediction, I overwrote these values using the formula: Episode_Length_minutes * 0.9554.

```python
df_train.filter(pl.col("Episode_Length_minutes_Decimal_Len") > 2)
```



https://www.kaggle.com/code/masaishi/decimal-digits-analysis-eda?scriptVersionId=236025089&cellId=7

## Data Leak 2: Abnormal Number_of_Ads values

Normally, Number_of_Ads ranges from 0-3. However, I found 7 instances where this value exceeded 3, and these abnormal values closely matched the Listening_Time_minutes. For these cases, I applied Number_of_Ads * 1.0588 after ensemble prediction.

```python
df_train.filter(pl.col("Number_of_Ads") > 3.0)
```



https://www.kaggle.com/code/masaishi/decimal-digits-analysis-eda?scriptVersionId=236025089&cellId=8

## Data Leak 3: Records with identical features have similar listening times

Since this competition used synthetic data (as mentioned in the description), I hypothesized that certain feature combinations would have consistent listening times. I systematically evaluated feature combinations to identify groups with low variability in listening times (low RMSE when predicting with group means). I will write how to calc it later.

The following four feature combinations showed significant LB improvement when overwriting predictions with group means:

```python
[
    "Host_Popularity_percentage-Guest_Popularity_percentage-Episode_Num",
    "Host_Popularity_percentage-Guest_Popularity_percentage-ELen_Int",
    "Publication_Day-Guest_Popularity_percentage-ELen_Int-HPperc_Int",
    "Guest_Popularity_percentage-Episode_Num-ELen_Int-HPperc_Int",
]
```




## Feature Engineering

I implemented numerous feature transformations including:

1. **Cyclical Day/Time Features** - Converting time-based features to sin/cos representations
2. **Ratio Features** - Creating meaningful ratios between existing features
3. **Integer/Decimal Separation** - Splitting numerical features into integer and decimal components
4. **Sentiment Features** - Encoding categorical sentiment data numerically
5. **Polynomial Features** - Creating squared and cubed versions of important variables

```python
df = df.with_columns(
      # Day features
      pl.col("Publication_Day").cast(pl_f_type).mul(2 * np.pi / 7).sin().alias("Day_sin"),
      pl.col("Publication_Day").cast(pl_f_type).mul(2 * np.pi / 7).cos().alias("Day_cos"),
      pl.col("Publication_Day").cast(pl_f_type).mul(4 * np.pi / 7).sin().alias("Day_sin2"),
      pl.col("Publication_Day").cast(pl_f_type).mul(4 * np.pi / 7).cos().alias("Day_cos2"),
      # Time features
      pl.col("Publication_Time").cast(pl_f_type).mul(2 * np.pi / 4).sin().alias("Time_sin"),
      pl.col("Publication_Time").cast(pl_f_type).mul(2 * np.pi / 4).cos().alias("Time_cos"),
      pl.col("Publication_Time").cast(pl_f_type).mul(4 * np.pi / 24).sin().alias("Time_sin2"),
      pl.col("Publication_Time").cast(pl_f_type).mul(4 * np.pi / 24).cos().alias("Time_cos2"),
      # Ratio features
      (pl.col("Episode_Length_minutes") / (pl.col("Number_of_Ads") + 1)).fill_null(0).alias("Length_per_Ads"),
      (pl.col("Episode_Length_minutes") / (pl.col("Host_Popularity_percentage") + 1)).fill_null(0).alias("Length_per_Host"),
      (pl.col("Episode_Length_minutes") / (pl.col("Guest_Popularity_percentage") + 1)).fill_null(0).alias("Length_per_Guest"),
      # Episode length features
      pl.col("Episode_Length_minutes").floor().alias("ELen_Int"),
      (pl.col("Episode_Length_minutes") - pl.col("Episode_Length_minutes").floor()).alias("ELen_Dec"),
      pl.col("Host_Popularity_percentage").floor().alias("HPperc_Int"),
      (pl.col("Host_Popularity_percentage") - pl.col("Host_Popularity_percentage").floor()).alias("HPperc_Dec"),
      # Sentiment features
      (pl.col("Episode_Sentiment") == "2").cast(pl.Int8).alias("Is_Positive_Sentiment"),
      pl.when(pl.col("Episode_Sentiment") == "2").then(0.75).otherwise(0.717).cast(pl_f_type).alias("Sentiment_Multiplier"),
      # Squared features
      (pl.col("Episode_Length_minutes") ** 2).alias("Episode_Length_squared"),
      (pl.col("Episode_Length_minutes") ** 3).alias("Episode_Length_squared2"),
  )

  df = df.with_columns(
      (np.sin(2 * np.pi * pl.col("Episode_Num") / 100)).alias("Long_Term_Cycle_Sin"),
      (np.cos(2 * np.pi * pl.col("Episode_Num") / 100)).alias("Long_Term_Cycle_Cos"),
      (pl.col("Episode_Length_minutes") * pl.col("Sentiment_Multiplier")).alias("Expected_Listening_Time_Sentiment"),
  )

  df = df.with_columns(
      (
          (pl.col("Episode_Length_minutes") - pl.col("Episode_Length_minutes").median()).pow(2)
          + (pl.col("Host_Popularity_percentage") - pl.col("Host_Popularity_percentage").median()).pow(2)
          + (pl.col("Guest_Popularity_percentage") - pl.col("Guest_Popularity_percentage").median()).pow(2)
          + (pl.col("Number_of_Ads") - pl.col("Number_of_Ads").median()).pow(2)
      ).alias("Diff_Squared")
  )
```

The most impactful feature was simply `pl.col("Episode_Length_minutes").floor().alias("ELen_Int")` - extracting the integer part of the episode length.

## Utilizing the Original Dataset

I incorporated this original dataset in two ways:

1. Direct concatenation with the competition training data; “Original Data as New Cols”
2. Adding specific rows as "Original Data as New Rows"

For the second approach, I was inspired by Chris Deotte's solution post from the Binary Prediction with a Rainfall Dataset competition (Playground Series - Season 5, Episode 3) shared [here](https://www.kaggle.com/competitions/playground-series-s5e3/discussion/571176).

The technique involved finding rows in the original dataset that matched the same feature group combinations I identified in my target encoding process. When I found matching feature groups, I added those original rows with their actual Listening_Time_minutes values as new training data. This essentially allowed me to enrich my dataset with additional ground truth values from the original source.

```python
for cols in combinations_list:
        n = f"pte_{'_'.join(cols)}"
        means = df_pltpd.group_by(cols).agg(pl.col("Listening_Time_minutes").mean().alias("mean_listening_time"))
        df = df.join(means, on=cols, how="left").with_columns(pl.col("mean_listening_time").fill_null(m).alias(n)).drop("mean_listening_time")
```

## Select Feature Combinations for Target Encoding

Code: https://www.kaggle.com/code/masaishi/select-feature-combinations-by-rmse

I'd like to share my approach for finding optimal feature combinations through target encoding. This method systematically evaluates different feature groupings to identify which combinations best predict podcast listening time.

Here's the step-by-step process I implemented:

1. First, split the dataset into training (80%) and validation (20%) sets:
2. Generate feature combinations to test (1-4 features at a time):
3. For each combination, create groups by concatenating feature values:
    
    ```python
    # For first feature in combination
    if comb[0] in df.select(cs.numeric()).columns:
        concat_expr = pl.col(comb[0]).round(round_num).cast(pl.Utf8)
    else:
        concat_expr = pl.col(comb[0]).cast(pl.Utf8)
    
    # Add remaining features to create group identifier
    for col_name in comb[1:]:
        if col_name in df.select(cs.numeric()).columns:
            concat_expr = concat_expr + "_" + pl.col(col_name).round(round_num).cast(pl.Utf8)
        else:
            concat_expr = concat_expr + "_" + pl.col(col_name).cast(pl.Utf8)
    ```
    
4. Calculate the mean listening time for each group in the training data:
    
    ```python
    df_train = df_train.with_columns(
        concat_expr.alias("group").cast(pl.Categorical)
    )
    
    df_group = df_train.group_by("group").agg(
        pl.col("Listening_Time_minutes").count().alias("count"),
        pl.col("Listening_Time_minutes").std().alias("std_listening_time"),
        pl.col("Listening_Time_minutes").mean().alias("mean_listening_time"),
        pl.col("Episode_Length_minutes").mean().alias("mean_episode_length"),
    )
    ```
    
5. Apply these group means to the validation set as predictions:
    
    ```python
    df_valid = df_valid.with_columns(
        concat_expr.alias("group").cast(pl.Categorical)
    )
    df_valid = df_valid.join(
        df_group[["group", "mean_listening_time"]],
        on="group",
        how="left",
    )
    ```
    
6. Calculate RMSE and other statistics for each combination:
    
    ```python
    std_distributes.append({
        "group": "-".join(comb),
        "cover_rate": df_group["count"].sum() / df_train["Listening_Time_minutes"].count(),
        "len": len(df_group),
        "mean_count": df_group["count"].mean(),
        "median_count": df_group["count"].median(),
        "std_count": df_group["count"].std(),
        "mean_std_listening_time": df_group["std_listening_time"].mean(),
        "rmse": calculate_rmse(df_valid["Listening_Time_minutes"], df_valid["mean_listening_time"]),
    })
    ```
    
7. Sort results by RMSE to find the best feature combinations:
    
    ```python
    std_distributes = std_distributes.sort("rmse")
    ```
    

This approach is essentially a systematic way to perform target encoding across different feature combinations. The best combinations provide valuable insights about which podcast attributes most strongly influence listening time.



## Model Ensemble

For my final submission, I created an ensemble combining several different regression models:

- HistGradientBoostingRegressor
- LGBMRegressor
- SVR (Support Vector Regression)
- TabNetRegressor
- XGBRegressor

## Experiment Management

Effective experiment management was crucial to my success in this competition. I tracked experiment parameters using Wandb, while implementing an automated commit system that included validation scores and Wandb experiment names in commit messages. This made it easy for me to identify which code changes produced the best results.





With over 1,000 experiments conducted throughout the competition, establishing this structured environment from the beginning proved invaluable. The screenshot examples show how my tracking system organized results visually.

I also carefully structured my code architecture to maximize flexibility:

```
src/
├── config.py
├── data
│   ├── data_class.py
│   ├── data_process.py
│   ├── default_selecteds.py
│   ├── encoders.py
│   ├── feature_eng.py
│   ├── simple_data_process.py
│   ├── simple_feature_eng.py
│   └── tabnet_feature_eng.py
├── main.py
├── models
│   ├── hgbr.py
│   ├── lgb.py
│   ├── svr.py
│   ├── tabnet.py
│   ├── test.py
│   └── xgb.py
└── utils.py
```

Each model followed a consistent interface with standardized input/output formats:

```python
def train_model(fold: int, datasetXy: DatasetXy) -> tuple[float, list | None]:
    # Model implementation

@dataclass
class DatasetXy:
    X_train: pl.DataFrame
    y_train: pl.Series
    X_valid: pl.DataFrame
    y_valid: pl.Series
    X_test: Optional[pl.DataFrame] = None
    y_test: Optional[pl.Series] = None
```

This design made it easy for me to test new models or apply successful feature engineering techniques across different algorithms. My configuration system also allowed for quick switching between prediction and evaluation modes.

While these details might seem minor, they significantly enhanced my ability to iterate efficiently throughout the competition.

## Special Thanks to:

- AngelosMar - https://www.kaggle.com/angelosmar1
- Chris Deotte - https://www.kaggle.com/cdeotte
- Panagiota Moraiti - https://www.kaggle.com/giotamoraiti
- Pranshu Bahadur - https://www.kaggle.com/pranshubahadur
- Spiritmilk - https://www.kaggle.com/act18l
- Ravi Ramakrishnan - https://www.kaggle.com/ravi20076
- Masaya Kawamata - https://www.kaggle.com/masayakawamata
- Chinmaya - https://www.kaggle.com/chinmayadatt
- Farukcan Saglam - https://www.kaggle.com/greysky
- Carl McBride Ellis - https://www.kaggle.com/carlmcbrideellis
- Thomas Meißner - https://www.kaggle.com/thomasmeine
