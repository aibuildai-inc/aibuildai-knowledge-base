# 1st Place Solution Team Isamu & Matt

Competition: ashrae-energy-prediction
Rank: #1
Source: https://www.kaggle.com/c/ashrae-energy-prediction/discussion/124709

Thank you to Kaggle and ASHRAE for hosting this competition. The decision to use only non-leak data for the private test set helped to make this competition fair. Thank you to all those who contributed to the kernels and discussions and especially those who made the leaks public. Last but not least, I'd like to thank and congratulate my teammate [Isamu Yamashita](https://www.kaggle.com/yamsam) for being a great teammate and becoming a Competitions Master.

During the competition, we shared ideas and discussed progress within our team, but we tested and trained separate models. This helped us maintain diversity in our final ensemble.  This is a combined summary our team's solution.  




## Preprocessing

### Remove anomalies
As others have noted, cleaning the data was very important in this competition.  The assumption is that there are unpredictable and hence unlearnable anomalies in the data that, if trained on, degrade the quality of the predictions. We identified and filtered out three types of anomalies:
1. Long streaks of constant values
1. Large positive/negative spikes
1. Additional anomalies determined by visual inspection

We noticed that some of these anomalies were consistent across multiple buildings at a site. We validated potential anomalies using all buildings in a site--if an anomaly showed up at the same time at multiple buildings, we could be reasonably certain that this was indeed a true anomaly. This allowed us to remove anomalies that were not necessarily part of a long streak of constant values or a large spike.

### Impute Missing Temperature Values
There were a lot of missing values in temperature metadata. We found that imputing the missing data using linear interpolation helped our models.

### Local Time Zone Correlation
As noted in the competition forum, the timezone in the train/test data was different from the timezone in the weather metadata. We used the information in this [discussion post](https://www.kaggle.com/c/ashrae-energy-prediction/discussion/112841) to correct the timezones.

### Target Transformations
Like most competitors, we started by predicting `log1p(meter_reading)`.  We also corrected the units for site 0 as per this [discussion post](https://www.kaggle.com/c/ashrae-energy-prediction/discussion/119261).  

Near the end of the competition, we tried standardizing `meter_reading` by dividing by `square_feet`; i.e., we predicted `log1p(meter_reading/square_feet)`.  Isamu came up with the idea after reading this [discussion post](https://www.kaggle.com/c/ashrae-energy-prediction/discussion/122263) by [Artyom Vorobyov](https://www.kaggle.com/artoby).  The models trained with the standardized target added diversity to our final ensemble and improved our score by about 0.002.  If we had more time we would have liked to explore this idea further; for example, we could have tried to predict `log1p(meter_reading)/square_feet` or created features using the standardized targets.

## Feature Engineering and Feature Selection
We took different approaches to feature engineering and feature selection in this competition.  Isamu took a conservative approach and carefully selected  features; on the other hand, Matt took a brute force approach and used most of them. Here are the features that helped:

* Raw features from train/test, weather metadata, and building metadata
* Categorical interactions such as the concatenation of `building_id` and `meter`
* Time series features including holiday flags and time of day features
* Count (frequency) features
* Lag temperature features similar to those found in the public kernels
* Smoothed and 1st, 2nd-order differentiation temperature features using Savitzky-Golay filter (see the figure below)



* Cyclic encoding of periodic features; e.g., `hour` gets mapped to `hour_x = cos(2*pi*hour/24)` and `hour_y = sin(2*pi*hour/24)`
* Bayesian target encoding (see this [kernel](https://www.kaggle.com/mmotoki/hierarchical-bayesian-target-encoding))

## Models
We trained CatBoost, LightGBM, and MLP models on different subsets of the data:
* 1 model per meter
* 1 model per site_id
* 1 model per (building_id, meter)

Our team tried different approaches to validation in this competition. Like other competitors, we tried K-Fold CV using consecutive months as the validation set.  The following code shows one approach to getting validation months: 
```
&gt;&gt;&gt; import numpy as np
&gt;&gt;&gt; def get_validation_months(n):
...     return [np.arange(i, i+n) % 12 + 1 for i in range(12)]
...
&gt;&gt;&gt; get_validation_months(6)
[array([1, 2, 3, 4, 5, 6]),
 array([2, 3, 4, 5, 6, 7]),
 array([3, 4, 5, 6, 7, 8]),
 array([4, 5, 6, 7, 8, 9]),
 array([ 5,  6,  7,  8,  9, 10]),
 array([ 6,  7,  8,  9, 10, 11]),
 array([ 7,  8,  9, 10, 11, 12]),
 array([ 8,  9, 10, 11, 12,  1]),
 array([ 9, 10, 11, 12,  1,  2]),
 array([10, 11, 12,  1,  2,  3]),
 array([11, 12,  1,  2,  3,  4]),
 array([12,  1,  2,  3,  4,  5])]
```
Trying different validation schemes allowed us to train models that added diversity to our final ensemble.

## Ensembling
To reduce the risk of overfitting to the public LB and improve robustness, we ensembled predictions from many different models. Here  are some of the things we did:
* Used cleaned leak data as a holdout set to tune our second stage model
* Averaged log values; i.e., `expm1(mean(log1p(x)))` rather than averaged the raw values
* Used the generalized weighted mean and tuned the parameters using Optuna
* Hedged our bets by including leak-free public kernels on cleaned data:
    * [ASHRAE: Half and Half](https://www.kaggle.com/rohanrao/ashrae-half-and-half) by [Vopani](https://www.kaggle.com/rohanrao)
    * [ASHRAE- KFold LightGBM - without leak (1.08)](https://www.kaggle.com/aitude/ashrae-kfold-lightgbm-without-leak-1-08) by [Sandeep Kumar](https://www.kaggle.com/aitude)
    * [Ashrae: simple data cleanup (LB 1.08 no leaks)](https://www.kaggle.com/purist1024/ashrae-simple-data-cleanup-lb-1-08-no-leaks) by [Robert Stockton](https://www.kaggle.com/purist1024)

This [kernel](https://www.kaggle.com/mmotoki/generalized-weighted-mean) shows how we ensembled our predictions.  Our final ensemble was a plain average of our top 4 submissions with respect to the public LB score. 

## What Didn't Work

* Again, following this [discussion post](https://www.kaggle.com/c/ashrae-energy-prediction/discussion/122263) by [Artyom Vorobyov](https://www.kaggle.com/artoby), we thought of ensembling the predictions for buildings with the same meter_readings; e.g., buildings 1225 and 1226 with meter 0. However, we were not able to improve our public LB result with this approach. It is possible that we did something wrong here, but we didn't have enough time to go back and explore this idea further. 

* Smoothing the predictions of our models.  We had initial success with smoothing the final predictions of our models, but after a certain point, we started to find that smoothing hurt our public LB score. Our guess is that smoothing helps spiky low quality predictions, but our ensemble predictions were already sufficiently smooth.
