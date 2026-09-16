# 2nd place solution - team JuneHomes (writeup)

Competition: amex-default-prediction
Rank: #2
Source: https://www.kaggle.com/c/amex-default-prediction/discussion/347637

*Note: Full code to retrain single model will be shared here in a 2 weeks.*

I would like to say thank you to competition hosts and Kaggle - it was a great pleasure to participate in tabular data competition after many months and years without one. 

Thank you to all participants (and of course winners - @daishu huge jump during last 3 days and fantastic solo result) - our success is your success - you forced us to try harder - without all of you It would be impossible to learn so many new things and achieve such result.

And special "thank you" goes to my fantastic teammates:
- Danila
- Alexey
- Igor

No words would be enough to say how much each of you contributed to the end result.
"Do Data Scientists have hobby? Yes -> DS competition".  

---

### Competition announcement

We were very hyped by the new tabular data competition release (sorry for the external link:  [link](https://www.linkedin.com/posts/konstantin-yakovlev-010b46125_american-express-default-prediction-kaggle-activity-6935250015619592194-hqnM)) and immediately decided to participate. Slack notification -> rules and perspective advertisement (100% chance too lose summer holidays and all free time) -> and here we are - four members. Only one member of the team had previous experience in DS competitions participation.

Few rules were established from the beginning:
- Only free time for competition
- No "second" accounts on Kaggle (even wife/friends to exclude any cheating suspicion)
- No competition discussion outside of the team
- We are here to learn and try our best

---

### Infrastructure and pipelines:

Each of us had own machines / resources (GCP/AWS/local). We used Kaggle platform just for a few times. So the first thing we wanted to solve - unified machine to save all artefacts / experiments. We decided to go with AWS. I would say that it is possible to achieve the same result that we have just with Kaggle resources, but it would be bit more stressful for team management. We didn't want to spend a lot of money on AWS but sometimes (during very hot hours) RAM spikes were 500GB+ to permit simultaneous work.

We tried to use neptune.ai for ML tracking but from July it was not very effective as we entered in brute force zone. 

*Advise: Resources management is very critical - find bottleneck and remove it to make your team most effective. At the same time don't burn money recklessly - limit your budged. If any optimization possible - do it as soon as possible to save time and resources.*

---

### Project Structure
Each run was internally versioned (ex. v1.1.1 - (major version).(fe version).(model version))
Overall project structure:
- Initial preprocess -> artifact cleaned and joined df
- FE -> Many aligned (by uid) dfs with separated features 
- Features selection -> dictionary we selection metadata
- Holdout Model (fe check and tuning) -> Local validation oof preds / holdout preds/ model / model metadata
- Full model run -> Model / Model metadata
- Prediction -> each fold oof predictions / cv split metadata / test predictions

All these permitted us to go back and forward and check what worked well and what did not and restore experiments in each particular step.

---

### Initial preprocess 
We wanted to achieve several things with this step:
- Join Train and Test -> due to many people involved I was afraid that some missed transformation on private test part will be unnoticed. So we sacrifice memory and speed optimization for overall stability and security.
- Remove detected noise -> (we had options here but ended with unified single one)
- Transform Customer ID to unified uid 
- Create internal subset feature -> Train / Public / Private
- Create unified kfold and holdout split -> To align all experiments
- Separate columns by type and store them separately to minify memory use and load time

#### Remove detected noise

We didn't use public notebooks for cleaning. Radar's Dataset is fantastic and it is 99% similar to our own transformations.
We used "isle" identification without any pre-build coefficients.
dummy code is something like this:

```
    for col in process_columns: 

        df = temp_df[[col]].sort_values(by=[col]) 
        df = df[df[col].notna()].drop_duplicates(subset=[col]).reset_index(drop=True)

        df['temp'] = np.floor(df[col] * 100000)
        df['group'] = ((df['temp'] - df['temp'].shift()).abs() >= 100).cumsum()

        i = 0
        while True:
            min_val = df[df['group']==i]['temp'].min()
            if min_val>0:
                break
            i += 1

        df['temp2'] = np.where(df['temp']>=0, 
                                np.floor(df['temp']/min_val).astype(np.int32),
                                np.round(df['temp']/min_val).astype(np.int32))
        
        mapping = dict(zip(df[col],df['temp2']))
        temp_df[col] = temp_df[col].map(mapping)

        print(col, df['group'].nunique(), df[col].nunique())
        print(df.groupby(['group'])['temp','temp2'].agg(['min','max','count','nunique']).head(40))
```

#### Create internal subset feature
We used last statement month to create 0/1/2 feature and store in in "index" df

#### Create unified kfold and holdout split
Fixed random seed (of course 42) to make spits and then took 20% of customers to holdout group (to test stacking / blending / etc)

#### Separate columns by type
After cleaning we had several columns "groups".
```
all_files = [
'p_columns', -> just p columns as we thought that they are very different (and P_2 is internal amex "scoring" model)
'objects_radar_columns', -> order encoding (we were checking where out cleaning differs from public approaches and here was the unique place) 
'objects_columns', -> onehot encoding
'categorical_cleaned__D__columns', -> no noise categoricals
'categorical_binary__S__columns', -> cleaned binary
'categorical_binary__R__columns', -> cleaned binary
'categorical_binary__D__columns', -> cleaned binary
'categorical_binary__B__columns', -> cleaned binary
'categorical__D__columns', -> removed noise categoricals
'categorical__B__columns', -> removed noise categoricals
'cleaned__B__columns', -> removed noise continuous 
'cleaned__D__columns', -> removed noise continuous 
'cleaned__R__columns', -> removed noise continuous 
'cleaned__S__columns', -> removed noise continuous 
'rest__B__columns', -> have no idea what to do with it -> floor 
'rest__D__columns', -> have no idea what to do with it -> floor 
'rest__R__columns', -> have no idea what to do with it -> floor 
'rest__S__columns', -> have no idea what to do with it -> floor 
]
```
Thanks again to @raddar we always used your preprocess as a baseline.

We were able to load just portion of data -> do fe -> concat to "index" as all dfs were aligned by index. Also such split permitted us to do fe by feature type to accelerate process and see more statistically valuable metric change. 

---

### FE

We started with careful fe column by column or small subset and it worked well until 1xx features and then any metric improvement or degradation was not statistically significant and many features "overlapped" on importance and significance. 

*Note: I believe that it is possible to build silver zone robust model with only 3xx features*

So we started from scratch with brute force))) Of course there was no need to apply "nunique" (for example to binary features) and our previous step helped us to limit fe.

```
all_aggregations = {
   'agg_func': ['last','mean','std','median','min','max','nunique'],
   'diff_func': ['first','mean','std','median','min','max'],
   'ratio_func': ['first','mean','std','median','min','max'],
   'lags': [1,2,3,6,11],
   'special': ['ewm','count_month_enc','monotonic_increase','diff_mean','major_class',
              'normalization','top_outlier','bottom_outlier','normalization_mean','top_outlier_mean','top_outlier_mean']
}    
```
+ pca (horizonal and vertical) + horizontal combinations + horizontal aggregations.

agg_func -> normal aggregations by uid
diff_func -> diff last - xxx -> std diff worked better than any other
ratio_func -> ratio_func last/xxx
lags -> diff last - Nx
special -> some special transformations -> count_month_enc worked well for categorical / emw for continous 

We ended up with about 7k features (stored file by group and by agg type for faster loading). 
Next thing was to figure out what works and what not -> this topic was the most challenging for us. 

#### Normalizations

It's better to call it Standardization (x - m) / s -> as we had also normalization test the name became constant "normalization")))
```
df.groupby(['dt_month','subset'])[col].agg(['mean','std'])
```
dt_month -> month of the statement
subset -> train / public / private
and mean and std from clients that had full statement history.

We have to have temporal shift to make it work. So we did a "trick" removed last statement for each client and applied exactly same transformation  for each client and merged appropriate labels. So we had 2 lines in training set for almost each client BUT validated results only on last statement during CV runs and Holdout checks. It more or less same as adding noised data but we had temporal drift and model was able to work better on unknown future data with  "possible" data drift.

---

### Features selection

Ooohh that was really fun. 

We used gbdt boosting type during experiments as it was very aligned with dart mode but was significantly faster.
Also, we used ROC AUC score during our experiments as we believed that due to amex instability we can't use it for decision making  (of course we tracked log loss and amex).

In previous step we brute forced many features and now is time to clean them out.
All feature selection was done with 5 CV folds training + independent check on 20% holdout data.

1. Zero importance -> Right from the start we were able to through away 1.5k features that had exactly 0 importance (lgbm importance). That means that with 250 bins and 2**10 data in leaf those features are not participating in any split.

2. Stepped hierarchical permutation importance -> we defined 300 initial features and looped over all other features subsets (600+) - was very time consuming but very stable.
Note: we shuffled order of the subset to force model try different combinations.
Add features subset -> train model -> permutate -> drop negative features (negative mean over 5 seeds) -> add new subset -> ...
During this part that took almost 3 days we limited features to 3k -> 0.800 lb

3. Stepped permutation importance.
Take all features -> train model -> permutate -> drop 20% of worst performed features (only negative) -> repeat. Final subset was 25xx features (and different from previous step) -> 0.800 lb

4. Forward feature selection.
We defined 300 initial features and simply added subset by subset and compared ROC AUC if metric change was > 0.0003 we kept the subset. -> 0.800 lb

5. Time series CV.
For very doubtful features as PCA and Normilized values we used to different validation stratagies:
- Train on first 6 month values (last statement of the first 6 months went to train set) and validate on last 6 (also just last statement of the last 6 months). We trained model without temporal feature and then with if result was better on CV and on holdout we added to final features subset.
- We used P_2, B_1, B_2 as a proxy target and MSE loss with combined Train and Test to see if we did right transformation and result did not degrade.

Many other options we tried but result was not stable.

Final subset came from "Forward feature selection" plus overlapped features from other technics minus overlapped negative combination.  -> lb 0.801 single model.

We tried to blend many models with different subset as we believed that it should give huge LB boost (based on holdout blending tests) but it didn't work well for lb. 

--- 

### Model 

In my own experience, DART never worked better and here we have proof that in DS "all depends." We did experiments with DART in the beginning and it did not show any metric improvement with our params and baseline model features subset. Later we found @ragnar123 notebook and gave it one more try and it worked marvelously.

From the beginning, we tried to build a more complex model with 2**7+ leaves and 0.7+ features but failed. It still puzzles me why a simple model with a very low number of features works here. 

I saw such behaviour mostly on synthetic data and stacking - so we tried to find out if data is syntetic (at least partly) and deanonimize internal scoring values - but didn't make it.

Our best single lgbm model was trained on 29xx features. 5 folds CV - no stratification by any option. Training data - 2 last staements for each client (transformed independently). Params:

```
lgb_params = {
    'boosting_type': 'dart',
    'objective': 'cross_entropy', 
    'metric': ['AUC'],
    'subsample': 0.8,  
    'subsample_freq': 1,
    'learning_rate': 0.01, 
    'num_leaves': 2 ** 6, 
    'min_data_in_leaf': 2 ** 11, 
    'feature_fraction': 0.2, 
    'feature_fraction_bynode':0.3,
    'first_metric_only': True,
    'n_estimators': 17001,  # -> 5000 for gbdt 
    'boost_from_average': False,
    'early_stopping_rounds': 300,
    'verbose': -1,
    'num_threads': -1,
    'seed': SEED,
}
```

Blend -> Power (2) rank blend of Dart lgbm (0.801 public) / GBDT lgbm  (0.799 public) / Catboost models (0.799 public)

Single lgbm with 3 last statements showed even better CV by we didn't have enough time to retrain it (full DART run for 5 folds took 12+ hours there).

It was obvious that clients with a little number of statements will not get benefit from all 2k features. So we created a special model that was trained only on 300 features with custom params (also dart). Predictions for clients with <=2 statements came exclusively from such model and were not blended with other models. 

How did we combine the result from 2 independent models to not destroy the final ranking?

| Client id | Number of statements | Basic ranking | <=2 prediction | Final ranking |
| --- | --- | --- | --- | --- |
| 1 | 13 | 5 | ... | 5 |
| 1 | 2 | 4 | 0.1 | 3 |
| 1 | 13 | 8 | ... | 8 |
| 1 | 1 | 3 | 0.5 | 4 |
| 1 | 13 | 7 | ... | 7 |
| 1 | 13 | 2 | ... | 2 |

we kept ranking for >2 statements and for rest resorted within initial ranking group (hope it's clear enough))). Was is perfect - no, but it was very stable with really tiny improvement (because of number of such clients in public and private test parts).

What also worked well:
- Train on all data without folds splitting and stop just 1000 rounds further then CV showed. 

What didn't work:
-  Stacking by any mean
-  Many models with different seed and fe order
- Massive blend of different models with different types and features (blend worked well till 0.799 and then any low performed model 0.795- made public score worse - we use public CV as additional holdout set and used approaches that worked well on local CV and LB - anything that worked partially was not used in the end).

Again, nothing really fancy here. The main thing that helped us align Train / CV with LB was very high  'min_data_in_leaf'.
No optuna used -> just manual old school tuning based on data feeling.
We did many experiments with weights and loss functions but none of them worked.

Due to AMEX metric specification it was obvious that focal loss should work but it didn't. We tried several times to switch loss function during the competition period and the result was the same. 

Error analysis showed that model makes errors without any "pattern" -> stacking didn't work for holdout set (25% of data)  and we had doubts that it will work on private/public test parts. We kept only LR/Lasso(0.02) for blending options to choose submissions.

Cross validation -> standard 5folds CV split by client ID. The unique thing that we did here is "prespliting" to align all CV between team members to be able to compare results directly.
 
---

What left without mentions:
- EDA on data
- Denoising experiments
- Data deanonymization -> didn't manage to make it
- Features pairs and triples combinations -> that didn't work well
- NaN filling -> didn't work
- Clusterization -> didn't work
- Hundreds of experiments with features selection process and internal discussions about it.
- Adding noised data (noise / swap noise) that leaded to interesting but doubtful results
- Model tuning 
- Removing absolute values and keep only diff or ratios -> should be more stable for future data but we saw some lb degradation and didn't proceed
- pseudo labeling

What we always wanted but didn't found time to do:
- NN - we have no NN in our final blend
- P_2 or any other column prediction (1/2/3/4 months ahead) with combined data and use it as meta information for lgbm main model
- 11 / 12 / 13 statements joined training on different subsets (df was too large and training was slow)

--- 

### Internal initial plan

```
########################### Data preprocessing and Data evaluation
#################################################################################

## Added noise removal -> GOOD2DO
# There is no doubt that some Noise was injected in data
# https://www.kaggle.com/competitions/amex-default-prediction/discussion/328514
# https://www.kaggle.com/competitions/amex-default-prediction/discussion/327649
# https://www.kaggle.com/competitions/amex-default-prediction/discussion/327651
# We need to find a way to remove it 
# the best option to not follow public approach
# At least with columns where columns have overlaped population

## Data minification for FE -> GOOD2DO
# Datatype downcasting
# Pickle/Parquet/Feather 
# Be careful with floats16 as it may lead to bad agg results
# Also float16 may lead to some signal degradation due to precision and values changes

## Evaluate values distributions and NaNs -> GOOD2DO
# Full 13 months history
# Train against Test Public and Test Private
# https://www.kaggle.com/competitions/amex-default-prediction/discussion/327926
#
# Need to try:
# Kolmogorov–Smirnov test -> GOOD2DO
#
# Adversarial validation -> GOOD2DO
# https://www.kaggle.com/code/carlmcbrideellis/what-is-adversarial-validation/notebook (just as simple example)
#
# Entropy / Distances / etc....
#
# Visual checks)))
#
# We need to find if ANY feature has very different distribution in PRIVATE test set
# If that feature works for Public part it doesn't mean that it will work for Private

########################### Targets
#################################################################################

# We need to find a way to get more targets -> GOOD2DO
# as we currently training on a single point by client we could greatly improve results
# by extending our training set with new targets
#
# Find default periods in current client history and make appropriate labeling -> GOOD2DO
# Make 2 level model -> predict p_2 values as normal time-series model and feed it to 2nd level GBT

########################### Separate Models
#################################################################################

# Probably it's a good idea to make separate models for each subdatasets
# Full history (13 months)
# Less than 13 months

########################### External Data
#################################################################################
# We can try to add "Consumer index" or any other independent temporal feature
# Will not work if we will not be able to expand targets and add temporal feature

########################### FE
#################################################################################

# We didn't make anything special here -> July
# AGGS (Stats by client)
# Rollings
# History length feature (not sure if it will help with Private Test)
# Should we correct statements dates and add NaNs?
# ReRanking categorical features by P_2 or Target
# Clusterization (4+ groups feature by feature)
# Count and Mean encodings for categorical features
# Features combinations (sum/prod/power) -> bruteforce
# PCA or any other dimension reduction by features groups

# We need to find if there is "connection" between clients in Train -> Public Test -> Private Test
# we have 458913 + 924621 -> 1383534 If I were AMEX I would export 1M clients (or other round number)
# so may be 384 534 Clients are overlaps

# Clip by 5 - 95 percentile

########################### Features Selection
#################################################################################
# Permutation importance (use all fold only!!!) -> recursive elimination (because of quantity of features -> 3-4 rounds with 0 and 50% negative drop) 
# SHAP
# Highly correlated features (.98+?)
# Forward selection (may take ages and due aggs may be not effective - probably by feature block) 
# Backward elimination (may take ages and due aggs may be not effective - probably by feature block) 

########################### CV
#################################################################################
# Mean Target differs my "history length" -> could be wise to do GroupedStratifeidFolds by history length
# For sure Splits should be done by client
# Target stratification to balance folds

########################### Loss function / Metric
#################################################################################
# Clean and fast np/torch metric
# Now it's in helper (need to cleanup that)
# https://www.kaggle.com/code/rohanrao/amex-competition-metric-implementations
#
# I don't believe that we will have better results with different loss function
# But it worth to try at least focal loss
# https://maxhalford.github.io/blog/lightgbm-focal-loss/
#
# Weights -> we should try change weights there
# weights by class
# weights by some history length
# weights by internal fe group
#
# We need custom metric for catboost
# example https://catboost.ai/en/docs/concepts/python-usages-examples#logloss1

########################### Models
#################################################################################

## First level choice
# LGB/XGB/CTB -> our main models here for sure
# After stabilizing the baseline model and base feature we need to make 1st round tuning

## Catboos specials
# Categorical features
# Embeding features

## NN (GPU/TPU) -> RNN / LSTM / Transformer
# TPU -> tensorflow (as it works better there)

## NN -> AE / VAE / DAE -> as a denoising model hidden layer as input for GBT models
# No need complex approach - just fast check the idea and in case of success move to big model

########################### Blending
#################################################################################
# Weighted Average
# Power Average
# Weighted Rank Average
# Linear/SVM
# Postprocessing?
```
