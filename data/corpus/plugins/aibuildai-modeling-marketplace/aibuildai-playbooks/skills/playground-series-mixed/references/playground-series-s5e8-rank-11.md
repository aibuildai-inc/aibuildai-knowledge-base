# 11th Place Solution: AutoGluon with Two-Feature-Set Blending

Competition: playground-series-s5e8
Rank: #11
Source: https://www.kaggle.com/c/playground-series-s5e8/writeups/11th-place-solution-autogluon-with-two-feature-set

This was my first serious attempt at a Kaggle competition, and I learned a lot from experimenting with different feature sets through feature engineering.
Eventually, I ended up with two different feature engineering pipelines and blended their predictions using AutoGluon.

The overall architecture of my solution can be visualized as follows:
.png?generation=1756703169336333&alt=media)

**Feature Engineering with EDA**
I tested each feature engineering idea using a simple CatBoost model with 5-fold CV:
- Quantile transformation
- qcut (binning)
- Binary features from day/amount columns (e.g., pdays_exist, balance_exist)
- Day column binning: grouped into ranges like >0, >=7, >=30, >=90, >=180, >=365
- Day/Amount column div & mod: created cyclical or periodic patterns (e.g., remainder after division)
- Arithmetic feature generation: ratios, differences, sums
- Numerical → categorical conversion
- Log-transform
- Target encoding

**Numerical to Categorical and Mistakes**
The feature I paid the most attention to was numerical-to-categorical conversion. Many other participants simply converted numerical features into categorical ones, and I noticed that this simple step could boost performance quite a lot (from 0.965 to 0.972 with a simple CatBoost).
So I tried categorizing features one by one, and here are the results:
.png?generation=1756704796643914&alt=media)
Two features turned out to have the most significant impact. Both had widely spread distributions, so I assumed they might be influenced by certain periodic patterns or specific values.
When I grouped these features by value and counted the target, I found that some values had much larger sample sizes than their neighbors, and in some cases, the targets were all 0 or all 1.

Based on this, I created a new feature by assigning labels depending on the target distribution (e.g., all 1 targets → 1, all 0 targets → 0, mixed → -1), which improved performance.(It was not used in the feature set, but I tested it only for validation purposes)

At first, I thought this kind of categorization might only work well on CV or the public leaderboard, so as an alternative I also tried using quotient and remainder transformations to capture periodicity. However, it turned out that the categorical transformation was effective even on the private leaderboard. Looking back, I think this mistaken assumption prevented me from running more experiments that could have improved the solution further.

**Modeling and Hyperparameter Tuning -> Autogluon**
In the beginning, I used three types of models: XGBoost, LightGBM, and CatBoost. I started from their default parameters and then performed hyperparameter tuning with Optuna. For initialization, I either used parameter suggestions from ChatGPT or borrowed parameters from other notebooks, and then fine-tuned them with Optuna.

After that, I built multiple models by combining different feature sets with different hyperparameters, and finally ensembled them using meta models and hill climbing.

While tuning hyperparameters, I felt that the performance wasn’t improving as easily as I expected. At that point, I decided to give AutoML a try and switched to AutoGluon. I was impressed by its predefined hyperparameter settings and the ease of distributed training.

I started with a configuration of 5 folds, 1 feature set, 1 stacking layer, and medium quality (denoted as 5,1,1,medium), and later expanded to (10,2,2,medium) and (8,2,2,best).

**Final Submissions**
For one of the submissions, I selected the (8,2,2,best) setup. To further test the effectiveness of categorization, I created a new feature set by removing some features from the original set and adding target encoding and categorical transformations, then trained it again with the (8,2,2,best) configuration.

Finally, I blended feature set 1 and feature set 2. However, since AutoGluon’s inference time was very long and I was running out of time, I couldn’t perform CV-based blending and had to go with blind blending instead. Ironically, this blind blending gave me the best performance.

**Results**
| Method| CV Score | Public LB | Private LB |
|--------|-----------|-----------|------------|
|Base Feature Set(Feature Set1) + CatBoost optuna|0.97441|0.97524|0.97510|
|Base Feature Set(Feature Set1) + LGBM optuna|0.97337|0.97422|0.97405|
|Base Feature Set(Feature Set1) + Blending 16 Models(Hill Climbing)|0.97556|0.97612|0.97600|
|Base Feature Set(Feature Set1) + Autogluon(5,1,1,medium)|0.9758|0.97660|0.97646|
|Base Feature Set(Feature Set1) + Autogluon(5,1,1,best)|0.97601|0.97667|0.97659|
|Base Feature Set(Feature Set1) + Autogluon(10,2,2,medium)|0.97578|0.97636|0.97624|
|Base Feature Set(Feature Set1) + Autogluon(8,2,1,best)|0.97640|0.97688|0.97685|
|* Base Feature Set(Feature Set1) + Autogluon(8,2,2,best)|0.97647|0.97693|0.97687|
|Categorical Feature Set(Feature Set2) + Autogluon(8,2,2,best)|0.97689|0.97746|0.97704|
|* Blind Blending(0.5, 0.5)(Feature Set1, Feature Set2)|---|0.97787|0.97766|

**What I Learned**
This competition gave me a great opportunity not only to try out an AutoML model for the first time, but also to revisit EDA, feature engineering, feature selection, hyperparameter tuning, and ensembling after a long time. Although I made some mistakes and have a few regrets, I was fortunate to end up with a good result. My next step is to apply what I learned from this competition to my work and aim for even better outcomes.
