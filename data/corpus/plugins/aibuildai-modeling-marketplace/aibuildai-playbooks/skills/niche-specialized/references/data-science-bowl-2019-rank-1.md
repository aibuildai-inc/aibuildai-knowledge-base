# 1st place solution

Competition: data-science-bowl-2019
Rank: #1
Source: https://www.kaggle.com/c/data-science-bowl-2019/discussion/127469

Thanks to Booz Allen Hamilton, Kaggle and everyone for this wonderful competition. And also thanks to my teammate @oyxuan . Congratulations to all winners!

# 1.  Summary
Our selected score is based on a single lightgbm (average on multi-seeds 5 fold). 

The model score : 
private qwk 0.568, public qwk 0.563
cv weighted qwk 0.591, cv weighted rmse 1.009

# 2.  Validation Strategy
In the early game , we find that the LB score is unstable and has low correlation with the local cv, so we decide to focus on the local cv score only. We have tried several ideas to make the local cv stable. Below are two validation sets we use:

**2.1 GroupK CV** : We use the 5 times * 5-fold GroupK by installation_id, each time with random groupk split seed and random column order. However qwk is still not so stable on our local cv, so we mainly concern the weighted rmse when validating our ideas and ignore qwk. For the weighted loss , the weight is the sample prob for each sample (We use full data, for the test part, we calculate the expectation of the sample prob as weight). 

**2.2 Nested CV**: Usually, the GroupK cv above works well. When we think the GroupK cv's decision has low confidence (eg. inconsistent with our common sense), we will use another nested set for double check:  We simulate the train-test split on the local data :  random select 1400 users with full history for the nested training and 2200 users with truncated history for the nested testing. We repeat it for 50~100 times and calculate the mean score for validation.

# 3. Feature Engineering
Most of our time are spending on feature engineer. We generate around 20,000 features these days, and use the [null importance method](https://www.kaggle.com/ogrellier/feature-selection-with-null-importances) to select the top 500 features. 

1. Lots of stats (mean/sum/last/std/max/slope) from true attempts ratio, correct true ratio, correct feedback ratio etc. Stats based on same assessment or similar game are highest important (Similar game : we map each game to the corresponding similar assessment, since they are similar task)

2. We extract features from different parts of the child history data : 1) full history part, 2) last 5/12/48 hour, 3) from last assessment to the current assessment. Since here are some shared devices phenomenon, add different part info may help model.

3. Event interval features (next event timestamps - current event timestamps) : Stats (mean/last) of event interval groupby event_id / event_code. Several event interval features show high importance.

4. Video skip prop ratio : clip event interval / clip length provided by organizer. (Does the child skip the video? If so, when does he skip?)

5. Event data feature : Stats(mean/sum/last) of all numerical args in event data X event_id / event code combination. We get the combination and args type from the specs file. 
eg. `event_code2030_misses_mean`.

# 4. Feature selection
1. Drop duplicate cols
2. Truncated adversarial validation to make sure there is no leak and no code errors, the mean adversarial AUC should be around 0.5.
3. Use [null important method](*https://www.kaggle.com/ogrellier/feature-selection-with-null-importances*) to select top 500 features.

# 5. Model
1. **Data augmentation** : The model is trained on the full data (full train history and test previous, improve + 0.002). 
2. **Loss** : We use rmse loss for training, and weighted rmse loss for validate. 
3. **Threshold** : Then use [Opitmizer Rounder](https://www.kaggle.com/naveenasaithambi/optimizedrounder-improved) to optimize thresholds for weighted qwk.
4. **Ensemble** : We just try a simple blending method (0.8 * lightgbm + 0.2 * catboost,  the private score is 0.570. Since the cv score is not improved, we do not select it for our final results.

#Thanks for reading!
