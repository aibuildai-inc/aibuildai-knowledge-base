# 44th Place Solution - Simple LGBM

Competition: godaddy-microbusiness-density-forecasting
Rank: #44
Source: https://www.kaggle.com/c/godaddy-microbusiness-density-forecasting/discussion/418355

Thank you for the competition on a very interesting topic. It was a time-series forecasting problem that predicts the true future, and was a fun theme to solve. I understood that the United States is a very large and diverse country.



This is a simple LightGBM, 44th Place Solution.

### Solution Overview

- Only LightGBM with 32 features　

- Multiplier Prediction
   I chose target as the multiplier between the previous month's data and the month to be predicted.

- Target smoothing
   Because there were so many outliers in the data, I smoothed the target by taking the medians of the three cases before and after the target.

- Model to forecast 1-6 months later
   I created several models to forecast 1-6 months later separately.

- Average of 3 months
   Average the multipliers for 3 months, including the months before and after the month of forecast. This is more accurate.

- Conversion of population from 2020 to 2021

- Round(0) since it is a discrete value

### My Notebook

I have published my notebook below.
[https://www.kaggle.com/code/thajime/godaady-44th-solution-lgbm](https://www.kaggle.com/code/thajime/godaady-44th-solution-lgbm)
