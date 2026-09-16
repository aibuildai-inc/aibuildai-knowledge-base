# 6th place solution

Competition: predict-energy-behavior-of-prosumers
Rank: #6
Source: https://www.kaggle.com/c/predict-energy-behavior-of-prosumers/discussion/499397

I'd like to thank Enefit and Kaggle for hosting this competition. It was a fun competition and I learned a lot from it. 

Final submission: https://www.kaggle.com/code/mmotoki/submit-blend-v5-4-0/notebook


## Features
Most of my features are the same or slightly transformed versions (to match the target) from the public notebooks. Thank you to everyone in the community who contributed to the public notebooks and discussions! The main difference in my features is that I also created simple baseline models for [consumption](https://www.kaggle.com/code/mmotoki/anyfit-consumption/notebook) and [generation](https://www.kaggle.com/code/mmotoki/anyfit-generation/notebook), which I retrained every day.  


## Models
I trained separate LightGBM models for production and consumption on 4 different targets (8 models total)

1. `target - target_lag48`
2. `target/(target_lag48_avg + 1) - baseline_pred`
3. `target/(target_lag48_avg + 1) - target_lag48`
4. `target/(target_lag48_avg + 1)`

where `target_lag48_avg` is the daily average of the 2-day lagged target. I didn't think to scale by `installed_capacity`, but from other top solutions, that seems like a good target. 

## Retraining
I actually didn't retrain my LightGBM models at all. The only retraining was in my baseline models. 


## Blending
My final model was a weighted combination of the models with different targets, 10 seeds, and two hyperparameter settings. In total, I had `160 models = 8 (targets) × 10 (seeds) × 2 (hyperparameter settings)`. In hindsight, I probably should have tried to allocate time to retraining LightGBM rather than creating such a big blend or retraining the simple baseline models so frequently.

| Target | Consumption Weight | Generation Weight |
| ---- | ---- | ---- |
| 1. | 0.13021 | 0.18193 |
| 2. | 0.49116 | 0.10890 |
| 3. | 0.19497 | 0.52331 |
| 3. | 0.19300 | 0.19148 |
