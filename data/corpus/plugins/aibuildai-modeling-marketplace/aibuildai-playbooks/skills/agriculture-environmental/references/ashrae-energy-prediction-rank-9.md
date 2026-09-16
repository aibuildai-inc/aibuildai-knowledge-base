# 9th place solution

Competition: ashrae-energy-prediction
Rank: #9
Source: https://www.kaggle.com/c/ashrae-energy-prediction/discussion/123525

Finally nice competition! Not easy due to really noisy data but new lessons learnt again. Some insights of my solution that reached place #9 and gold.

One of my objectives was to survive to shake-up. As soon as leaked data were discovered I decided to use it mainly for hold-out validation. I started training without any leak data, playing with features engineering, different models, different time CV folds and drove my work only with **correlated CV + hold-out + LB results**. It was a bit frustrating to have very low ranking with this strategy but I knew it should help at the end. On the 2 last week I included 2017 leak data in training and kept 2018 leak data for hold-out.

My solution is **ensemble (ridge regression) of several models**:
- LightGBM (x7)
- CatBoost (x4)
- Neural Network with [categories embedding](https://www.kaggle.com/isaienkov/keras-nn-with-embeddings-for-cat-features-1-15#685556) and features standard normalization (x4)
- LiteMORT (x1)

**Features are quite simple**, no model above 15 features, 12 as average:
- `building_id`, `meter`, `site_id`, `primary_use`, `week_day`, `is_holiday`
- `square_feet`, `cloud coverage`, `precip_depth_1_h`
- [feels like](https://github.com/malexer/meteocalc) temperature, building age, `square_feet` * `floor_count`
- `air_temperature` roll mean 24h, `sea_level_pressure` trend roll mean 24h
- `air_temperature` [cooling degree](https://www.investopedia.com/terms/c/colddegreeday.asp) per day, `meter_reading` median per building per meter per year

Notice that some of my models are not using building_id to try to generalize better.
A few models are per meter, others not.
For each model CV I applied different time split, x3, x4 and x6 .

**Cleansing and imputation was important too**. For weather data, I tried to find different external  sources to fill gaps but it did not give any boost mainly because we're not 100% sure of location of `site_id` and some data such as cloud coverage was not consistent with the ones in training data. Finally, I trained an additional simple LightGBM model to impute missing data based on provided data (similar to this [kernel](https://www.kaggle.com/frednavruzov/nan-restoration-techniques-for-weather-data)). For meter reading, cleansing was not obvious, removing zero patterns (electricity, hot water in summer ...) looked a good idea but one can notice that such pattern also appear in 2017/2018 leaked data. Some buildings could also have some renovations slots that would explain zero patterns. So each of my model had a zero-pattern drop strategy different (`site_id` = 0 before May 2016 only, full drop, partial drop based on duration and/or on season).

**Post-processing**: None and it was a mistake when I see top solutions.

For final submission I selected the ones with best CV/Hold-Out/LB correlation and it was a good choice as they're the best score in Private LB!

**What did not work:**
- Too many features lead to overfit especially with target encoding.
- Tree-based second level model for ensembling (overfit again)
- External data for weather (not in inline with provided data)
- Non time-split CV (auto-correlation)

Thanks to organizers, Kaggle and competitors for this challenge!
