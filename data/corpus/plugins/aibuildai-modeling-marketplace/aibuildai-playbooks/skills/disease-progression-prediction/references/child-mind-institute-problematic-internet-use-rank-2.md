# 2nd Place Writeup

Competition: child-mind-institute-problematic-internet-use
Rank: #2
Source: https://www.kaggle.com/c/child-mind-institute-problematic-internet-use/discussion/552712

I was really surprised to secure 2nd place in this competition, especially since I had essentially quit a few months earlier. My [final approach](https://www.kaggle.com/code/aradhyeagarwal/starter-notebook-with-polars-gpu) was based on a fork of the [Starter Notebook](https://www.kaggle.com/code/onodera/starter-notebook-with-polars-gpu). The key observation was that the original notebook didn’t explicitly handle missing values, and it relied on CatBoost to do it automatically. While I wasn’t deeply familiar with CatBoost’s internal approach, I felt that leveraging some domain knowledge through a custom imputation strategy might be a strong alternative.

Below is the dictionary I used to handle missing values in different columns:


```python
replacement_strategy = {
    'Basic_Demos-Age': 'average',
    'Basic_Demos-Sex': 'new_number',
    'CGAS-CGAS_Score': 'average',
    'Physical-Diastolic_BP': 'average',
    'Physical-HeartRate': 'average',
    'Physical-Systolic_BP': 'average',
    'Fitness_Endurance-Max_Stage': 'average',
    'Fitness_Endurance-Time_Mins': 'average',
    'Fitness_Endurance-Time_Sec': 'average',
    'FGC-FGC_CU': 'new_number',
    'FGC-FGC_CU_Zone': 'new_number',
    'FGC-FGC_GSND_Zone': 'new_number',
    'FGC-FGC_GSD_Zone': 'new_number',
    'FGC-FGC_PU_Zone': 'new_number',
    'FGC-FGC_SRL_Zone': 'new_number',
    'FGC-FGC_SRR_Zone': 'new_number',
    'FGC-FGC_TL_Zone': 'new_number',
    'BIA-BIA_Activity_Level_num': 'average',
    'BIA-BIA_Frame_num': 'average',
    'SDS-SDS_Total_Raw': 'average',
    'SDS-SDS_Total_T': 'average',
    'PreInt_EduHx-computerinternet_hoursday': 'average'
}
```

I didn’t leave everything to CatBoost because it can’t **fully “understand”** the meaning behind each feature. For instance, the number of internet usage hours might appear as discrete integers, but we know it has a clear ordering and a specific interpretation (more hours = higher usage). Other features, like IDs of a certain item, also appear as discrete integers but don’t carry such a natural ordering. Only we, as humans, can make that distinction and apply suitable imputation methods.

 Replacement Rules:

- Average: If a feature is numeric (float or integer) and requires an averaged value, the missing entries are replaced by the mean of all existing values.
- New Number: If a feature is integer-based (often treated as categorical) and we want to keep it distinct, we take the current max value in that column and replace missing entries with (max + 1). If the feature is a string-based category, we replace missing entries with "Null".

I only used these strategies if the column was strictly an integer (or recognized as categorical). When a feature was float-based, the “average” replacement was more reasonable, assuming it held a numeric meaning rather than discrete categories.
I also increased the number of folds for cross-validation from 5 to 20. After that, I saw diminishing returns, so I stopped. Ironically, despite the improved cross-validation score, my public leaderboard score was lower than the baseline notebook, which was quite unexpected. The final private leaderboard result, however, came as a pleasant surprise. I’d love to hear any ideas on why there’s such a significant difference between the public and private scores in the context of my approach.

Cheers,
Aradhye
