# 1st Place - RAPIDS cuML Stack - 3 Levels!

Competition: playground-series-s5e4
Rank: #1
Source: https://www.kaggle.com/c/playground-series-s5e4/discussion/575784

Thank you Kaggle for a great playground competition. This month's playground competition has a great dataset with lots of interesting patterns within! (and it felt more like real data and less like synthetic data). And there are lots of strong competitors which made this competition fun and exciting!

The target `Listening_Time_minutes` is approximately equal to the linear relationship `0.72 x Episode_Length_minutes` as described in my 3 discussion posts [here][1], [here][2], [here][3]. The other 9 features modulate this linear relationship. Based on this insight, I stacked the following approaches:



My favorite solution for a Kaggle competition is a **single model**, my second favorite solution is **hill climbing ensemble**. Neither of these solutions could win 1st place in Kaggle's April playground competition because the data has too many interactions and too many deep patterns. For Kaggle's April playground competition, we need a large diverse deep **RAPIDS cuML stack of 3 levels**!

# Hill Climbing (linear level 2 model) versus Stacking (non-linear level 2 model)
Hill climbing (or ridge) ensemble generally works well. However in this competition, the dataset was so complicated that a deep stack was the best solution. The most important feature is `Episode_Length_minutes`. It contains **90%+** of the signal. But it is missing for **11.6%** of the data! This means there are two scenarios; 
* Predict target `Listening_Time_minutes` **with** `Episode_Length_minutes` 
* Predict  `Listening_Time_minutes` **without** `Episode_Length_minutes`

Hill climbing (and ridge) cannot do this (because it uses a linear level 2 model). Imagine that we make one model that does great predicting target **with** ELM and we build a second model that does great predicting target **without** ELM. Hill climbing will just take a weighted average of all predictions. 

But a stack (non-linear level 2 model) will use predictions from one model when predicting **with** ELM and use the predictions from another model when predicting **without** ELM. In other words, instead of taking all predictions from all models, it will take the best predictions from each model (for different situations)!

# RAPIDS cuML Stack - 3 Levels of Models!
The secret to building a strong stack is diverse models. (And every model trains with the **same** 5 KFolds and we **must** remove all leaks from target encoding, pseudo labeling, etc). Diversity comes from different feature engineering and different models (and/or model hyperparameters). 

For each new model I built, I engineered different sets of features using the speed of **RAPIDS cuDF**. Each model has different customized features that benefit the new model best. And I trained lots of diverse models using the speed of **RAPIDS cuML**! All models below use the speed of **GPU**!



# Diversity x5
To add diversity to our stack we can take each of the 12 model depicted above and train it in at least  5 different ways described below. Additionally, we can change feature engineering and/or hyperparameters and train more ways. My final stack uses **75 models**. So I approximately created each of the above 12 models in 6 different ways!

Every day during the month of April, I spent a few hours and built new diverse models. Using **3xA100 GPU** and the speed of **RAPIDS cuDF and cuML** I would build about a dozen new models (with new complex feature engineering) each day and keep the few models which improved my stack!

### (1) Different Sets of Feature Engineering
The typical way to predict `Listening_Time_minutes` is to train a model using KFold and all columns of `train.csv`. Additionally we can create more columns with feature engineering. We can build multiple GBDT models each using different engineered features. This provides diversity to our stack. Also we can change GBDT hyperparameters. For example, some times we use `max_depth=10` and sometimes we use `max_depth=0, max_leaves=1024`. These find different interactions and create diverse models. Furthermore, sometimes we can use `max_depth=20` to get more interaction and sometimes `max_depth=5` for less interaction. Below are 4 other ways to train models in April's playground competition.

### (2) Remove Episode_Length_minutes from All Rows!
Based on my discussion [here][2], the feature/column `Episode_Length_minutes` is important. We can remove `Episode_Length_minutes` **from all rows** and train a model to predict `Listening_Time_minutes` from all other columns. These models will be strong predicting target when `Episode_Length_minutes` is missing. And the stack will use these models when appropriate.

### (3) Predict Ratio of Target divided by Episode_Length_minutes
Based on my discussion [here][1], for each model, we can create a new target with `train['new_target'] = train.Listening_Time_minutes / train.Episode_Length_minutes`. We can train models to predict this new target. We can then multiply this prediction by `Episode_Length_minutes` or an imputed value of `Episode_Length_minutes` from below.

### (4) Predict Episode_Length_minutes (use Train.csv and Test.csv)
Based on my discussion [here][2], the feature `Episode_Length_minutes` is so important, we can train models to predict `Episode_Length_minutes`. Futhermore, we can use both `train.csv` and `test.csv` data to train and predict `Episode_Length_minutes`. Because both `train.csv` and `test.csv` have all the columns necessary. 

Afterwards, we can use these ELM predictions in at least 3 ways. (1) We can impute missing values with these ELM preds then train a model. (2) We can replace every row's ELM (both missing and non-missing) with these ELM preds, then train a model. (3) We can multiply these ELM preds by the Ratio preds (from above) to predict the target `Listening_Time_minutes`. All 3 of these ideas will make new diverse models! 

### (5) Pseudo Label (use Train.csv and Test.csv)
Based on my discussion [here][3], we see that many columns are important. We can use more information from more columns by using the columns from `test.csv`.  We can add `test.csv` data with pseudo labels to the training of all our models.

# Stacking Models CV Scores
Below are the CV scores for level 1, level 2, and level 3 models (without pseudo labeling). The LB scores are basically the same as the CV scores:

| **Level 1 Model** | Notes | CV Score |
| --- | --- | --- |
| RAPIDS cuML Lasso | uses 6000 features! | 13.2 |
| RAPIDS cuML SVR | uses 6000 features! | 13.2 |
| RAPIDS cuML KNN Regressor | k=51, weight by distance | 12.8 |
| RAPIDS cuML Random Forest | max_depth = 32 | 12.1 |
| NN - MLP | Built by ChatGPT | 12.0 |
| NN - TabPFN | 20x "SUBSAMPLE_SAMPLES": 10_000 | 13.2 |
| GBDT - XGBoost | 4x models with 4x feature sets | 11.8 |
| GBDT - LGBM | diverse from XGBoost | 11.8 |
| GBDT - Boost over RAPIDS Lasso | predict Lasso residuals | 11.9 | 
| GBDT - Boost over RAPIDS SVR | predict SVR residuals | 11.9 | 
| GBDT - Boost over NN MLP | predict MLP residuals | 11.9 | 
| AutoML AutoGluon | public notebook [here][4] | 12.4 |
| --- | --- | --- |
| **Level 2 Model** | **Notes** | **CV Score** |
| GBDT XGBoost | uses 73 level 1 models | 11.56 |
| NN - MLP | uses 73 level 1 models | 11.56 |
| --- | --- | --- |
| **Level 3 Model** | **Notes** | **CV Score** |
| Weighted Average | 50% / 50% | 11.54 |

.
**CREDITS:** Thank you @pirhosseinlou for XGBoost single model [here][5] and @greysky for LGBM single model [here][6] which I used as two of my XGBoost "4x models with 4x feature sets" (and then made a dozen variations of). And thank you @itasps for your AutoML AutoGluon model [here][7]. I incorporated all 3 of these public models into my final stack (by re-running with my stack's KFolds and then making a dozen variations of each)!

# Final Submission - CV 11.54, Public LB 11.51, Private 11.44, **First Place!**
My final **RAPIDS cuML** stack has CV 11.54, Public LB 11.50, Private 11.44, **First Place!**

# Post Comp Analysis
Now that the comp ended, I compare Hill Climb to Stack with my 73x L1 models:
* Hill Climbing - CV 11.64 - Public LB 11.57 - Private LB 11.503 
* Stack - CV 11.54 - Public 11.51 - Private 11.448 

[1]: https://www.kaggle.com/competitions/playground-series-s5e4/discussion/573002
[2]: https://www.kaggle.com/competitions/playground-series-s5e4/discussion/574249
[3]: https://www.kaggle.com/competitions/playground-series-s5e4/discussion/571549
[4]: https://www.kaggle.com/code/itasps/automl-autogluon-podcast
[5]: https://www.kaggle.com/code/pirhosseinlou/xgboost-single-model
[6]: https://www.kaggle.com/code/greysky/ps-s5e4-lgbm-cv-12-25-lb-12-15
[7]: https://www.kaggle.com/code/itasps/automl-autogluon-podcast
