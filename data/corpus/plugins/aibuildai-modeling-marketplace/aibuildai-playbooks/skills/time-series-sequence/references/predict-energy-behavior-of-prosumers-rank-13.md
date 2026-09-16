# 13th place solution

Competition: predict-energy-behavior-of-prosumers
Rank: #13
Source: https://www.kaggle.com/c/predict-energy-behavior-of-prosumers/discussion/499364

## 13th place solution

First and foremost, I would like to express my gratitude to the organizers for hosting this competition and extend congratulations to all the winners. Here is a summary of the solution that secured the 13th place.

## Overview
I developed separate models for production and consumption using LightGBM. Instead of using the raw target values, for production, I used the difference of the value two days prior divided by installed capacity. For consumption, I used the difference of the value two days prior divided by the eic count. I retrained the models three times during the inference period.

## Validation
I used a single fold validation for the period from January 2023 to May 2023.
When submitting, I retrained on all data with two seeds for both production and consumption models.

## Target
Production: (target - target_lag2) / installed_capacity
Consumption: (target - target_lag2) / eic_count

## Main Features
- Production:
    - Date related features (e.g., holidays, months, weekdays)
    - eic_count, installed_capacity
    - Lag and differences of target (production) / installed_capacity, including statistical measures
    - Lag and differences of target (production) / eic_count, including statistical measures
    - Lag and differences of target (consumption) / installed_capacity, including statistical measures
    - Lag and differences of target (consumption) / eic_count, including statistical measures
    - Weather forecasts
    - Difference between weather forecast and historical weather.
    - Weather forecasts 2 hours before, 1 hour before, 1 hour after, and the average from 1 hour before to 1 hour after
    - Averages of lagged target (production) / installed_capacity for weekdays and weekends (e.g., the average of the last two weekdays if the prediction day is a weekday, and the last two weekends if it's a weekend)
    - https://www.kaggle.com/competitions/predict-energy-behavior-of-prosumers/discussion/468654 thanks [@mohammadpakdaman0](https://www.kaggle.com/mohammadpakdaman0)

- Consumption
    - Date related features (e.g., holidays, months, weekdays)
    - Lag and differences of target (production) / installed_capacity, including statistical measures
    - Lag and differences of target (consumption) / installed_capacity, including statistical measures
    - Lag and differences of target (consumption) / eic_count, including statistical measures
    - Weather forecasts
    - Lag of historical weather 
    - Averages of lagged target (consumption) / eic_count for weekdays and weekends (e.g., the average of the last two weekdays if the prediction day is a weekday, and the last two weekends if it's a weekend)
- For computational efficiency, only the top 300 most important features were used in the production model and top 400 in the consumption model as determined by LightGBM.

## Re-training
Re-training was conducted three times: at the end of January 2024, February 2024, and March 2024. At each of these times, the models for both production and consumption were trained with two seeds each.
