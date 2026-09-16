# ~1st Place Solution LGBM with some adjustments

Competition: covid19-global-forecasting-week-5
Rank: #1
Source: https://www.kaggle.com/c/covid19-global-forecasting-week-5/discussion/155638

## Summary
LGBMs with quantile regression were trained on time-series and geo features for short term predictions. Aggressive 1w average smoothing was used for long term predictions. Due to the large number of small locations the top 30 country/state had to be adjusted manually.

#### **Notebooks**
https://www.kaggle.com/gaborfodor/c19w5-create-submission
https://www.kaggle.com/gaborfodor/c19w5-check-submission
https://www.kaggle.com/gaborfodor/c19w5-train-lgbs

## Feature Extraction

* Population
* Latitude, Longitude
* Day of week
* Share of total cases for each day of week
* Rolling mean/std for 1w, 2w, 3w
* Cumulative totals
* Confirmed - Fatality rate
* Trend from last 2-3 weeks
* Normalized features by population
* Nearby features based on the closest 5-10-20 locations


Rescaled and rounded features to 1-2 decimals to decrease overfitting

### External data
I started with searching public us county level demographic data (age, income, population density, covid19 lockdown info etc.) I found a few useful sources though I did not have time to clean and merge them. The only external data I used was the geo encoded lat-lon coordinates for each location.

## Modeling

For each target/quantile/forecast lag separate model was trained 
with location based 5-fold CV and early stopping based on pinball loss.
Models were only trained to predict the next 1-14 days.

- Trained bunch of LGBMs with random parameters to blend
- Sample Weighting based on location weights and time decay 


## Post processing
- Clipped negative predictions at 0 
- Made sure the 0.05 (0.95) quantile predictions are not higher (lower) than the median
- Smooth daily predictions (`Y[t] * 0.66 + Y[t-1] * 0.33`)
- For US country total used the state level rollups for median
- Manually inspected and adjusted the top 30 countries
- Flat long-term predictions based on the last predicted weekly average
- Small daily decay was added to 0.05 quantile and median



