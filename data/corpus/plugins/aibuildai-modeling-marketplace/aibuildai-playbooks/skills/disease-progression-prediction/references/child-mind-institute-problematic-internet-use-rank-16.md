# 16th Place Solution

Competition: child-mind-institute-problematic-internet-use
Rank: #16
Source: https://www.kaggle.com/c/child-mind-institute-problematic-internet-use/discussion/552569

It may have been partly due to luck, but I'm happy to have made a shake-up into the gold medal range.

I published my solution notebook.
- https://www.kaggle.com/code/rsakata/cmi-piu-16th-place-solution

The main points of my solution are as follows:
- imputation of missing values with ItrativeImputer
- feature engineering from parquet files
- LightGBM training with custom QWK objective and metric
- performing 10 x 10 nested cross-validation to get reliable validation scores and stable test predictions
- performing threshold optimization only once using the overall predictions from the nested cross-validation.

Please refer to my notebook for more details on the above points.
To confirm the robustness of my solution, I changed the seed of StratifiedKFold and checked the LB scores in late submissions. The results are as follows.

| Seed  | Public LB | Private LB |
| --- | --- | --- |
|0| 0.441 | 0.468 |
|1| 0.442 | 0.466 |
|2| 0.446 | 0.464 |
|3| 0.432 | 0.472 |
|4| 0.447 | 0.465 |
|5| 0.435 | 0.470 |
|6| 0.432 | 0.471 |
|7| 0.442 | 0.470 |
|8| 0.441 | 0.470 |
|9| 0.446 | 0.469 |
| **ave.** | **0.440** | **0.469** |  

The performance seems to be relatively stable. I also conducted a brief ablation study as follows.

| Description  | CV (nested) | Public LB | Private LB |
| --- | --- | --- | --- |
| Original | 0.4884 | 0.433 | 0.470 |
| without Parquet Features | 0.4821 | 0.442 | 0.464 |
| without Missing Value Imputation | 0.4726 | 0.423 | 0.438 |
| without Parquet Features and Missing Value Imputation | 0.4602 | 0.440 | 0.412 |
| without Custom Objective and Metric | 0.4810 | 0.436 | 0.471 |

It is based on the results of a single execution, but judging from these result, it seems that missing value imputation was important in this competition. Unfortunately, my custom objective and metric only contributed to the CV score. However, I plan to continue exploring the effectiveness of this idea.

Thanks for reading!
