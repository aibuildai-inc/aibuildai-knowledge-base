# 13th place gold solution

Competition: ashrae-energy-prediction
Rank: #13
Source: https://www.kaggle.com/c/ashrae-energy-prediction/discussion/123496

Thanks to ASHRAE for providing this unclean dataset! Thanks to Kaggle for continuing and keeping the competition alive. Congrats to everyone who managed to survive the LB shakeup!

Thanks to @rohanrao, @kailex, @nz0722, @aitude, @purist1024 for your excellent notebooks which had direct impact on helping me achieve this outcome.

## Summary

Final 2 submissions: equal weighted blend of the following plus some regularization.
1. https://www.kaggle.com/purist1024/ashrae-simple-data-cleanup-lb-1-08-no-leaks
2. https://www.kaggle.com/rohanrao/ashrae-half-and-half (credit for original work https://www.kaggle.com/kailex/ac-dc)
3. https://www.kaggle.com/aitude/ashrae-kfold-lightgbm-without-leak-1-08
4. https://www.kaggle.com/nz0722/aligned-timestamp-lgbm-by-meter-type

public LB: 0.935 (1.032 w/o leak estimated) with aggressive regularization 0.80 and 0.91
public LB: 0.944 (1.039 w/o leak estimated) with conservative regularization 0.91 for all
public LB: 0.950 (1.045 w/o leak) no tricks - I did not select for submission

### What worked
1. Data Cleaning - garbage in, garbage out. This is probably the single most important aspect of the the competition. I did this manually by plotting [heatmaps](https://www.kaggle.com/ganfear/missing-data-and-zeros-visualized) and also going into each building's meter to inspect the target meter readings if they looked reasonable. I also reverse engineered the heat map to show only zeros by added the following line: <br> `train_df = train_df.query('not (meter_reading != 0)')`
2. Regularization - (this is what I'm calling it, maybe some will call it postprocessing, coefficients, tricks, etc.) - multiplying by some value &lt; 1.0. For aggressive regularization, I used two different values 0.80 for responsive meters and 0.91 for less responsive meters. I probed each site's meter individually by multiplying by 0.95 to start then as I went through all the meters, I noted which meter or meters dropped the public LB by 0.001, which I noted to be responsive to regularization). I got the idea from [LANL Earthquake prediction](https://www.kaggle.com/c/LANL-Earthquake-Prediction/discussion/94324). Fortunately, I was confident that the LB would not shake me down if I tried to overfit the LB this way. I ran some tests on the leaked sites 0,1,2,4,15 to test each site's meters effect to various values. 0.90-0.95 seemed fairly safe values to start out with when I began probing. I gained about 0.015 on LB through a series of 29 submissions, which took about 2 weeks. As my remaining submissions diminished, I ramped up the overall aggressiveness of regularization and began probing the responsive site's meters as a collective submission because there wasn't enough submissions to try all possible values. This gave between 0.004 - 0.005 reduction on Private LB score. Without any regularization, I would have ended up placing 35 on private LB, so this trick definitely gave my scores the extra kick to finish in gold.
3. Feature Engineering - In the half-and-half model, I added a feature that grouped `building_id`, `meter`, `weekday`, `hour` and mean target encoded it using full train (after data cleaning). I got the idea from this [unassuming kernel](https://www.kaggle.com/mlisovyi/no-ml-benchmark). In Kfold-lightgbm-without-leak-1-08 model, I added a feature that combined site and meter as a categorical with no mean encoding. I noticed that for some sites, the mean encoded `bm_week_hour` feature performed worse while others performed better, but overall, it seemed favorable.
4. Validating using sites 0,1,2,4,15 - using actual test ground truths for various sites individually and together helped to monitor whether my experiments improved out of sample test data.
5. Addition by Subtraction - removing certain features for certain models helped improve ground truth (GT) test validation as well as local cross validation (CV). For half-and-half model variant, I dropped `site_id`, `sea_level_pressure`, and `precip_depth_1_hr`. For kfold-lightgbm-without-leak-1-08 I dropped `site_id`, `sea_level_pressure`, `wind_direction`, `wind_speed`, `year_built`, `floor_count`. I removed holiday features for all existing models because they made CV and GT validation worse. For aligned-timestamp-lgbm-by-meter-type, I dropped all the lag3 features.


Notes: very little hyper-parameter optimization was performed. Just very limited basic tuning.

### What didn't work
1. Training using leaked test GT labels - training with GT from 2017-2018 did not improve out of sample `site_id`. For ex: training with `site_id` 0 did not improve validation scores for sites 1,2,4,15 dramatically. I only performed that one test and realized that adding a `site_id` to testing doesn't improve LB scores for out of site validation because each `site_id` is it's own microcosm and behaves different from other sites.
2. A lot of feature engineering did not work including weather features.
3. CatBoost/XGBoost/NN Embeddings/Linear Regression/Kernel Ridge Regression(KRR)/KNN all performed worse. I tried building 1449 (1 for each building) and 2380 (1 for each building's meter) Linear Regression, KRR, and LGBM models. I tried blending with Catboost and NN Embeddings models separately, but GT validation didn't seem to improve.
4. Training with full year of training data and validating using sites 0,1,2,4,15 didn't seem to help much.
5. Correcting site 0 meter 0 didn't help - [discussion](https://www.kaggle.com/c/ashrae-energy-prediction/discussion/119261)

### Some experiments I tried
1. Using half-and-half methodology to split train into 2 separate halves - middle months (4-9)/ending months (1-3, 10-12) → performed worse. First half of the day 0-11, later half of the day 12-23. This split by hour in the day performed surprisingly well and GT validation had shown this, however I didn't select it for final submission blend. [notebook](https://www.kaggle.com/teeyee314/best-single-lgbm-lb-1-08-morning-evening)
2. L2 model - I stacked a model on top of individual model predictions (for the same folds), which only worked for half-and-half model. I left this out of final blend. I also tried ensembling through stacking final models using LGBM but there didn't seem to be any improvement in GT validation.

### What I didn't try
1. [Divide and Conquer notebook](https://www.kaggle.com/rohanrao/ashrae-divide-and-conquer) - I didn't bother with this notebook at all so I can't tell whether it was any good. Looking back now, I should have at least tried playing around with it. It is possible it could have helped reduce variance like @rohanrao mentioned.
2. Different blending methods, different weights, etc. - I just kept it simple (introduced no additional complexity or bias)
3. Making submission with certain models in final blend due to limited submissions. Namely I wanted to submit my blend with NN Embeddings, but based on GT validation, I filtered out a lot of potentially better performing blends. It is possible that sites 0,1,2,4,15 did not represent other sites well enough, but that was the trade off I had to make.  

notes: I mention these, because it may have been the difference for finishing closer in-the-money. 

### What I learned
1. I took my mean target encoding game to a new level with this competition. Before this, I was stuck at basic single categorical feature mean target encoding. Below is the code I used to do multi categorical feature mean encoding:
&gt; bm\_cols = ['building\_id', 'meter', 'weekday', 'hour',]
df\_train['hour'] = df\_train['timestamp'].dt.hour
df\_train['weekday'] = df\_train['timestamp'].dt.weekday
bm = df\_train.groupby(bm\_cols)['meter\_reading'].mean().rename('bm\_week\_hour').to\_frame()
df\_train = df\_train.merge(bm, right\_index=True, left\_on=bm\_cols, how='left')

### Final submissions used for blending
[half-and-half variant](https://www.kaggle.com/teeyee314/best-single-lgbm-lb-1-08)
[kfold-lightgbm-without-leak-1-08 variant](https://www.kaggle.com/teeyee314/kfold-lightgbm)
[aligned-timestamp-lgbm-by-meter-type variant](https://www.kaggle.com/teeyee314/aligned-timestamp-lgbm-by-meter-type-1-09)

### Final thoughts
It took me a year to finally reach a gold model. I was an absolute beginner when I started and I've learned a lot since joining Kaggle. I hope to continue competing and learning from all the bright and inspiring people on here. Although the kernels I blended with are all public, there were a lot of details that led to obtaining a gold model. Most notably, stable validation and logging each experiment meticulously so that after a few weeks or near the end, I wouldn't forget what led to improvements and what did not.
