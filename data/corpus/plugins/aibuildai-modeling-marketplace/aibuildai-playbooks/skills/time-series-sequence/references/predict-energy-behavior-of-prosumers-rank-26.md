# 26th place solution

Competition: predict-energy-behavior-of-prosumers
Rank: #26
Source: https://www.kaggle.com/c/predict-energy-behavior-of-prosumers/discussion/499475

Thanks to kaggle and everyone involved for hosting such an exciting competition.
Submission notebook is [here](https://www.kaggle.com/code/flat831/private-26th-code).
I'm referring to [vitalykudelya’s excellent notebook](https://www.kaggle.com/code/vitalykudelya/enefit-update-submission-logic).(Big thanks!!)

# Overview
Weighted ensemble of LightGBM models with different targets (objective = "regression_l1").
The models are divided into production and consumption based on is_consumption.

# Validation Strategy
I used a single fold validation for the period from Feb 2023 to May 2023.

# Target
* production(weights for ensemble)
  * target - target_48hr(0.15)
  * target - target_mean_5days(0.1)
  * target - target_mean_7days(0.25)
  * target / istalled_capacity(0.5)

* consumption(weights for ensemble)
  * raw target(0.3)
  * target - target_48hr(0.1)
  * target - target_168hr(0.25)
  * target - target_mean_5days(0.1)
  * target - target_mean_7days(0.25) 

target_mean_5days = mean(target_48hr ~ target_120hr)
target_mean_7days = mean(target_48hr ~ target_168hr)

# Features
Basically, my feature is from [this Notebook](https://www.kaggle.com/code/vitalykudelya/enefit-object-oriented-gbdt).
* I’ve excluded those derived from historical weather.
* I've added a few more, but there weren't any significant effects.

# Re-training
Re-training was conducted only once: at the end of January 2024.
