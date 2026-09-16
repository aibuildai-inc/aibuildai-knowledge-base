# 3rd Place - Target Encoding and 3 Levels

Competition: playground-series-s5e4
Rank: #3
Source: https://www.kaggle.com/c/playground-series-s5e4/discussion/575862

Thanks for a fun and interesting competition - the dataset was large enough for meaningful CV but still small enough to work comfortably on standard local hardware. Congrats to @cdeotte for yet another exceptional performance, and to @greysky for an impressive second place with a single (!!!) LGBM model!

My approach relied heavily on Target Encoding and a large ensemble of models with diverse features and hyperparameters.

During most of the competition, I used weighted averaging via Nelder-Mead or Hill Climbing. As a last-minute experiment, I tried stacking - something that’s never worked for me in a Kaggle competition - which got me an immediate boost from 11.66 to 11.62 CV on the very first and only try. I should have explored and optimized it earlier. I suspect stacking worked better here because of the high percentage of missing values in the single most important feature.

In the end, I blended my stacking ensemble (80%) with my best scoring Hill Climbing ensemble (20%), effectively turning this into a three-level approach.

All models were trained with 5-fold CV (simple KFold). I believe going for 7 or 10 folds could have slightly improved the score, but the tradeoff in runtime didn’t seem worth it for me.

I did some quick hyperparameter tuning with Optuna, but interestingly, the final ensemble benefitted from including some older, non-optimized models. Diversity seems to have worked especially well in this competition.

**Model Overview**
Level 1:
- 10 × LGBM (various features and params)
- 5 × XGB
- 4 × CatBoost
- 2 × RandomForest
- 1 × ExtraTrees
- 4 × HistGradientBoostingRegressor

Level 2:
- Hill Climbing
- LGBM

Level 3:
- Weighted Averaging

Interestingly, HistGradientBoostingRegressor consistently received negative weights (-0.08 to -0.21) during Hill Climbing but were never discarded. Does anyone have an explanation for this? Other models (linear models) were rejected in Hill Climbing.

**Best Single Model CV Scores:**
- LGBM: 11.79
- XGB: 11.81
- CatBoost: 11.93
- ExtraTrees: 11.96
- HistGB: 11.99
- RandomForest: 12.05

**What Worked Well**
- Target Encoding: This was the backbone of most of my feature engineering. I experimented with median, min, max, nunique but mean TE scored best.
 - I concatenated 2- to 7-grams, converting all columns to string before. 
 - I dropped TE features with a very high or very low cardinality.
 - To speed things up, I saved TE features as parquets per fold. This forced me to use the same KFold settings the whole competitions to avoid leaks. 
 - My top model used about 270 TE features.
 - I used a custom TE implementation (no smoothing, no fillna), though using the Scikit-learn version would likely have yielded similar results.
- Fixing extreme outliers for Number_of_Ads and Episode_Length_minutes helped, see https://www.kaggle.com/code/stopwhispering/podcast-eda
- Adding number of decimal digits of string-formatted Episode_Length_minutes as a feature. Thanks @AngelosMar for that interesting finding. Adding as a feature worked slightly better than manually correcting predictions with high number of decimal digits. 
- Some Divide/Minus/Plus/Combine interaction features 
- Original Dataset: For the original dataset I went for the concatenation approach this time, just adding original data to the train dataset. For CV scoring I ignored the original dataset records. I also added a flag 'is_original'.

**What Didn’t Work**
- Predicting ratio (Target / Episode_Length_minutes) instead of target with a separate model for Episode_Length_minutes being NaN
- Any kind of linear model or NN
- Imputation
- Binning, Clustering
- Label encoding of categorical variables

**Feature Selection**
For feature selection (that is mostly Target Encoding this time) I used a very greedy variant of Sequential Feature Selection using an LGBM with high learning_reate and low n_estimtors: After scoring all candidates, I selected the best scoring ~10-20 and started again - far from optimal but my local hardware didn't allow for more sophisticated selection. I tried some Recursive Feature Elimination afterwards, but that simply took too long and didn't yield good results. I did, however, remove features with extremely high correlation to other features.


Updates: added some details on TE, original dataset, and feature selection.
