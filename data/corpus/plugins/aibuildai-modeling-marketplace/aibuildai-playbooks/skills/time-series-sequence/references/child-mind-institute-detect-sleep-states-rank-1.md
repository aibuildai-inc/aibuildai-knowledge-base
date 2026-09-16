# 1st place solution

Competition: child-mind-institute-detect-sleep-states
Rank: #1
Source: https://www.kaggle.com/c/child-mind-institute-detect-sleep-states/discussion/459715

First of all, I would like to express gratitude to all participants and the competition host. It was a challenging competition, but I am pleased with the positive outcome and feel relieved.

Here is a brief summary of our solution.
You can check our code [here](https://github.com/sakami0000/child-mind-institute-detect-sleep-states-1st-place).

## Single model

Here's a log on how to improve the CV score after the summary. The final scores were: CV: 0.8206, public LB: 0.768, private LB: 0.829 (equivalent to 9th place).

### Summary

#### Model structure

The model structure is primarily based on [this amazing notebook](https://www.kaggle.com/code/danielphalen/cmss-grunet-train), with a structure comprising:
CNN (down sample) → Residual GRU → CNN (up sample)



- SEScale

For input scaling, SEModule was utilized. (https://arxiv.org/abs/1709.01507)
```python
class SEScale(nn.Module):
   def __init__(self, ch: int, r: int) -> None:
       super().__init__()
       self.fc1 = nn.Linear(ch, r)
       self.fc2 = nn.Linear(r, ch)

   def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
       h = self.fc1(x)
       h = F.relu(h)
       h = self.fc2(h).sigmoid()
       return h * x
```

- Minute connection

As noted in several discussions and notebooks, there was a bias in the minute when ground truth events occurred. To account for this, features related to minutes were concatenated separately in the final layer.

```python
   def forward(self, num_x: torch.FloatTensor, cat_x: torch.LongTensor) -> torch.FloatTensor:
       cat_embeddings = [embedding(cat_x[:, :, i]) for i, embedding in enumerate(self.category_embeddings)]
       num_x = self.numerical_linear(num_x)

       x = torch.cat([num_x] + cat_embeddings, dim=2)
       x = self.input_linear(x)
       x = self.conv(x.transpose(-1, -2)).transpose(-1, -2)

       for gru in self.gru_layers:
           x, _ = gru(x)

       x = self.dconv(x.transpose(-1, -2)).transpose(-1, -2)
       minute_embedding = self.minute_embedding(cat_x[:, :, 0])
       x = self.output_linear(torch.cat([x, minute_embedding], dim=2))
       return x
```

#### Data Preparation

Each series of data is divided into daily chunks, offset by 0.35 days.

```python
       train_df = train_df.with_columns(pl.arange(0, pl.count()).alias("row_id"))
       series_row_ids = dict(train_df.group_by("series_id").agg("row_id").rows())

       series_chunk_ids = []  # list[str]
       series_chunk_row_ids = []  # list[list[int]]
       for series_id, row_ids in tqdm(series_row_ids.items(), desc="split into chunks"):
           for start_idx in range(0, len(row_ids), int(config.stride_size / config.epoch_sample_rate)):
               if start_idx + config.chunk_size <= len(row_ids):
                   chunk_row_ids = row_ids[start_idx : start_idx + config.chunk_size]
                   series_chunk_ids.append(series_id)
                   series_chunk_row_ids.append(np.array(chunk_row_ids))
               else:
                   chunk_row_ids = row_ids[-config.chunk_size :]
                   series_chunk_ids.append(series_id)
                   series_chunk_row_ids.append(np.array(chunk_row_ids))
                   break
```

During training, half of each chunk is used in every epoch.

```python
               sampled_train_idx = train_idx[epoch % config.epoch_sample_rate :: config.epoch_sample_rate]
```

For evaluation, overlapping sections are averaged, and the ends of each chunk are trimmed by 30 minutes.

#### Target

A decaying target is created based on the distance from the ground truth event, with diminishing values as the distance increases.



```python
tolerance_steps = [12, 36, 60, 90, 120, 150, 180, 240, 300, 360]
target_columns = ["event_onset", "event_wakeup"]

train_df = (
   train_df.join(train_events_df.select(["series_id", "step", "event"]), on=["series_id", "step"], how="left")
   .to_dummies(columns=["event"])
   .with_columns(
       pl.max_horizontal(
           pl.col(target_columns)
           .rolling_max(window_size * 2 - 1, min_periods=1, center=True)
           .over("series_id")
           * (1 - i / len(tolerance_steps))
           for i, window_size in enumerate(tolerance_steps)
       )
   )
)
```

The target is updated each epoch to decay further.

```python
               # update target
               targets = np.where(targets == 1.0, 1.0, (targets - (1.0 / config.n_epochs)).clip(min=0.0))
```

By attenuating the target, the range of predicted values narrows, allowing for the detection of finer peaks.



#### Periodicity Filter

As discussed in [here](https://www.kaggle.com/competitions/child-mind-institute-detect-sleep-states/discussion/456177), there is a daily periodicity in the data when the measuring device is removed. This is leveraged to predict these periods rule-based and used as a filter for inputs and predictions.



#### Input Features

- categorical features
  - hour
  - minute
  - weekday
  - periodicity flag
- numerical features
  - anglez / 45
  - enmo.log1p().clip_max(1.0) / 0.1
  - anglez, enmo 12 steps rolling_mean, rolling_std, rolling_max
  - anglez_diff_abs 5 min rolling median

### Change logs

- baseline model (cv: 0.7510) - public: 0.728
- Add a process to decay the target every epoch (cv: 0.7699, +19pt)
- Add a periodicity filter to the output (cv: 0.7807, +11pt)
- Add a periodicity flag to the input as well (cv: 0.7870, +6pt) - public: 0.739
- batch_size: 16 → 4, hidden_size: 128 → 64, num_layers: 2 → 8 (cv: 0.7985, +11pt) - public: 0.755
- Normalize the score in the submission file by the daily score sum (cv: 0.8044, +6pt)
- Remove month and day from the input (cv: 0.8117, +7pt)
- Trim the edges of the chunk by 30 minutes on both sides (cv: 0.8142, +4pt) - public: 0.765
- Modify to concatenate the minute features to the final layer (cv: 0.8206, +6pt) - public: 0.768

----

## Post Processing

This post-processing creates a submission DataFrame to optimize the evaluation metrics. With this post-processing method, we significantly improved our scores (public: 0.768 → **0.790**, private: 0.829 → **0.852** !!!).

This was a complex procedure, which I will explain step by step.

1. **Data Characteristics**

  First, let's discuss the characteristics of the data. As noted in several discussions and notebooks, the second of the target events was always set to zero.
  The competition's evaluation metric doesn't differentiate predictions within a 30-second range from the ground truth event. So, whether the submission timestamp's seconds are 5, 10, 15, 20, ... 25, the same score is returned.

  

2. **Creation of the 2nd Level Model**

  The 1st level model's predictions were trained to recognize events within a certain range from the ground truth as positive. However, the 2nd level model transforms these into probabilities of a ground truth event existing for each minute.
  The output of the 1st level model was at seconds 0, 5, 10, ..., but the 2nd level model aggregates these to always be at second 0. Specifically, it inputs aggregated features around hh:mm:00 and learns to predict 1 only at the exact time of an event, otherwise 0. Details of the 2nd level model will be described later.

  

3. **Score Calculation for Each Point**

  As explained earlier, submitting any second within the same minute yields the same score. Therefore, we estimate the score at the 15 and 45 second points of each minute, and submit the one with the highest value, effectively submitting the highest score for all points. The method of score estimation is as follows:

  For instance, let's estimate the score at 10:00:15.

  

  First, we create a window of 12 steps from the point of interest and sum the predictions of the 2nd level model within this window to calculate the `tolerance_12_score`.

  

  Similarly, we calculate `tolerance_36_score`, `tolerance_60_score`, ..., for the respective tolerances used in the evaluation, and the sum of these scores is considered the score for the point of interest.

  

  We perform this calculation for all points, and for each series, we adopt the point with the highest score and add it to the submission DataFrame.

  

4. **Score Recalculation**

  Next, we recalculate the score to determine the next point to be adopted. For example, suppose the point 09:59:15 was chosen.

  First, consider updating the `tolerance_12_score`. Events within tolerance 12 of the adopted point cannot match overlappingly with the next point to be submitted.

  

  Therefore, when calculating the `tolerance_12_score` for the next point to be adopted, it's necessary to discount the prediction values within 12 steps of the currently adopted point.

  

  Likewise, for `tolerance_36_score`, `tolerance_60_score`, ..., we recalculate by discounting the prediction values within 36, 60, ..., steps of the adopted point.

  

  With the updated scores calculated, we again adopt the highest scoring point for each series and add it to the submission dataframe.

  

5. **Creating Submissions**

  We repeat the above step 4 to extract a sufficient number of submission points, then compile these into a DataFrame to create the submission file.

### Additional Techniques

Several other techniques were employed to make the post-processing work effectively:

- Normalize the predictions of the 2nd level model daily.
- When recalculating the score, calculate the difference from the previous score to reduce the computation.
- Speed up the above calculations using JIT compilation.

### Details of the 2nd Level Model

- The 2nd level model starts by averaging the 1st level model's predictions on a per-minute basis and then detecting peaks in these averages using `find_peaks` with a height of 0.001 and a distance of 8.
- Based on the detected peaks, chunks are created from the original time series, capturing 8 minutes before and after each peak. (Recall: 0.9845)
  - This step_size was crucial because the ratio of positive to negative examples changes depending on how many steps are included, affecting the accuracy of subsequent stages. Therefore, we tuned the number of steps for optimal performance.
  - If chunks are connected, they are treated as a single chunk.



- For each chunk, we aggregated features from the 1st model's predictions and other features like anglez and enmo. These aggregated features were then used to train models such as LightGBM and CatBoost.
- Additionally, we treated each chunk as a sequence for training CNN-RNN, CNN, and Transformer models. As a result, we developed a model that could account for minute biases not fully addressed by the 1st level model.



- The predictions of the 2nd level model were sufficiently calibrated, so there was no need for further transformation.


