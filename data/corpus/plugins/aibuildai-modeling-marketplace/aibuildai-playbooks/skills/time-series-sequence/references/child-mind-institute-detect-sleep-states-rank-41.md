# 41st Place Solution for the Child Mind Institute - Detect Sleep States Competition

Competition: child-mind-institute-detect-sleep-states
Rank: #41
Source: https://www.kaggle.com/c/child-mind-institute-detect-sleep-states/discussion/461154

First, I would like to express my sincere thanks to the organizers of the Child Mind Institute - Detect Sleep States Competition. This competition provided an invaluable opportunity to engage with long one-dimensional sequences, offering a distinct and educational experience in an important area of data science.


## **Overview of the Inference Pipeline**
Our training pipeline and models are based on the excellent code by [@213tubo](https://www.kaggle.com/tubotubo), available at [this page](https://www.kaggle.com/competitions/child-mind-institute-detect-sleep-states/discussion/452940?rvi=1). 
#### **Our solution is as follows**:

- **Data Segmentation:**
  - Segmented input features using overlapping time windows for inference
- **Models:**
  - Utilized 10 Unet-based models, each trained on two distinct training-validation splits across five unique models.
  - Each of the 5 models has a different backbone.

- **PostProcessing(Single Model):**
  - Applied Non-Maximum Suppression (NMS) to retain only the highest-scoring prediction when detected peaks were within a specified proximity.

- **Ensemble Predictions:**
  - Adopted an original ensemble method inspired by Weighted Boxes Fusion (WBF)
  - In the first stage, models within the same fold were assembled.
  - In the second stage, these results from different folds were combined.

- **Post Processing(Ensembled Results):**
  - Removal of predictions in periodic padding intervals.
  
Below is an overview of our solution, as illustrated in the following diagram.



## Details of the Submission

### Models Used:
We trained Unet-based models featuring five different CNN backbones: ResNet101, MiT-B3, MiT-B5, ResNeXt101, and ResNet152.
Our models were trained using two different combinations of training-validation data, with approximately 20% of the training set allocated as validation data in each combination.

For each training session of the models, we varied the seed values to increase their diversity. This approach was employed to enhance the reliability of cross-validation (CV) results and to enrich the ensemble for submission.

### Data Segmentation for Inference:
For the input data segmentation, we used a specific window size and set the hop size to half of this window size, creating overlapping segments for inference. This method allowed us to effectively utilize the central part of each window in the final analysis, thereby reducing false detections at the edges of the segments.

### Ensemble Method
We used an original ensemble method inspired by Weighted Boxes Fusion (WBF). The algorithm steps include:

- Grouping the input dataframe by series_id and event.
Sorting each event by its score in descending order.
- Extracting other events within a certain range from different dataframes.
- Calculating the weighted step as sum(step*score) and calculating the new score as score/(number of dataframes)
- Removing events within the specified range but with lower scores.
#### Our Code:
```python
def ensemble_predictions(df_list: list, series_ids: list,step_threshold: int=50) -> pd.DataFrame:

    grouped_list = [df.groupby(['event', 'series_id']) for df in df_list]
    events = ["onset", "wakeup"]
    ensemble_predictions = []
    for series_id in series_ids:
        for event in events:
            sorted_group_list = []
            for grouped in grouped_list:
                if (event, series_id) in grouped.groups:
                    sorted_group_list.append(grouped.get_group((event, series_id)).sort_values(by='score', ascending=False))
            

            while any([not sorted_group.empty for sorted_group in sorted_group_list]):
                specific_index = next((index for index, sorted_group in enumerate(sorted_group_list) if not sorted_group.empty), None)
                highest = sorted_group_list[specific_index].iloc[0]
                ensemble_steps = [highest["step"]]
                ensemble_probas = [highest["score"]]
                probas_sum = highest["score"]
                highest_step = highest['step']

                for i, sorted_group in enumerate(sorted_group_list):
                    compared_prediction = sorted_group[sorted_group['step'].sub(highest_step).abs() <= step_threshold]
                    condition = sorted_group['step'].sub(highest_step).abs() > step_threshold
                    filtered_group = sorted_group[condition]
                    sorted_group_list[i] = filtered_group
                    if i != specific_index:
                        if not compared_prediction["score"].empty:
                            first_score = compared_prediction["score"].iloc[0]
                            first_step = compared_prediction["step"].iloc[0]
                        else:
                            first_score = 0 
                            first_step = 0
                        ensemble_steps.append(first_step)
                        ensemble_probas.append(first_score)
                        probas_sum += first_score
                
                ensemble_step = 0
                if probas_sum!=0:
                    for s, p in zip(ensemble_steps, ensemble_probas):
                        ensemble_step += s*(p/probas_sum)

                ensemble_predictions.append({"series_id": series_id, "step": int(ensemble_step), "event": event, "score": probas_sum/len(grouped_list)})

    ensemble_df = pd.DataFrame(ensemble_predictions)
    ensemble_df['row_id'] = ensemble_df.index
    return ensemble_df

```
### Post Processing
Post-processing steps were implemented for both single models and ensembled results.

- **For single models:**
  - Applied Non-Maximum Suppression (NMS) using the excellent code from [this discussion](https://www.kaggle.com/competitions/child-mind-institute-detect-sleep-states/discussion/458822) to retain only the highest-scoring prediction when detected peaks were within a specified proximity.

- **For ensembled models:**
  - Removal of needless predictions in periodic padding intervals was inspired by [this notebook](https://www.kaggle.com/code/takanashihumbert/a-simple-way-trying-to-find-the-dark-zones).


## Ablation Study
The details of the scores are as follows.
| Number | Description                                                               | CV (Fold1) | Public LB | Private LB | 
|--------|---------------------------------------------------------------------------|------------|-----------|------------|
| (1)    | ResNet101 (No overlap)                                                    | 0.756      | 0.722     | 0.748      |
| (2)    | ResNet101 (overlap)                                                       | 0.77       | 0.732     | 0.778      |
| (3)    | ResNet101 (overlap) + NMS                                                 | 0.773      | Not Submitted | Not Submitted |
| (4)    | Ensemble of 5 models (overlap, same fold) + NMS                           | 0.7844     | 0.753     | 0.790      |
| (5)    | Ensemble of 5 models (overlap, same fold) + NMS + Remove Predictions in periodic padding       | **0.7847**     | **0.756** | 0.801      |
| (6)    | Ensemble of 5 models(overlap) across 2 folds + NMS                                      | -          | 0.749     | 0.792      |
| (7)    | Ensemble of 5 models(overlap) across 2 folds + NMS + Remove Predictions in periodic padding            | -          | 0.753     | **0.802**  |
