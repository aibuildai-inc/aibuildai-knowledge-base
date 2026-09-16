# 1st place solution

Competition: playground-series-s3e7
Rank: #1
Source: https://www.kaggle.com/c/playground-series-s3e7/discussion/390976

**Feature Engineering**
Although all of the data was numeric, a look at the data description revealed that the features type_of_meal_plan, room_type_reserved, and market_segment_type were actually categorical in nature. I handled these variables in 3 different ways:
1. Leaving them as is
2. One-hot encoding them
3. Marking them as categorical features (used for LGBM)

I played around with creating additional features, but I didn't find anything that improved my CV significantly. 

**Data Leakage**
Within the train and test set, there were 1531 pairs of records that were duplicates if you dropped booking status. 
1. 562 of those pairs were in the train set, and upon closer inspection, each of those pairs had opposite booking statuses. 
2. 253 of those pairs were in the test set. 
3. The remaining 716 pairs included 1 record in the train set and 1 record in the test set. 

Out of curiosity, I tried modifying a submission by setting the predictions for those 716 records in the test set to the opposite of their corresponding record in the train set, and got a boost of +.014 to my leaderboard score. I naively hoped to keep this secret to myself, but it was [quickly discovered](https://www.kaggle.com/competitions/playground-series-s3e7/discussion/388851) by the rest of the community 😅 However, I do have to thank @siukeitin for their suggestion to predict .5 for the 253 pairs of duplicates in the test set. This boosted my score by +.003, and I don't believe this trick was as well-known.

Additionally, I removed from the train set the 562 duplicate pairs, plus the 716 records that had a duplicate in the test set, as I found that doing so improved my score on a holdout set I constructed to test this. I think this only resulted in a minor increase in my private score though. However, I do think that perhaps removing the train duplicate pairs helped make my CV more reliable.

**Data Cleaning**

When running adversarial validation between the train and original datasets, I plotted for the original dataset the predicted probability of belonging to the train set and got a bimodal distribution like this:


This led me to experiment with removing the chunk of the original dataset that looked the least like the train set. I ended up using models that both included the entire original dataset and models that removed ~17% of the original dataset that looked the least like the train set.

I also played around with things like fixing dates, but didn't see significant improvement in CV score from doing so.

**Modeling**
My CV setup was a stratified 3-fold with 3 repeats. For creating submissions, predictions from each fold were averaged. When tuning hyperparameters, one thing I noticed was that increased tree expressiveness seemed to improve performance to a much greater degree than what I had been used to. This meant using XGBoost with the 'exact' algorithm instead of 'hist' and max_depths around 12-13, and using LGBM with larger max_leaves and max_bin values.

My winning submission was an average of the following 4 XGB submissions and 2 LGBM submissions:
- XGB
- XGB with categorical variables one-hot-encoded
- XGB with reduced original dataset
- XGB with categorical variables one-hot-encoded and reduced original dataset
- LGBM with reduced original dataset
- LGBM with categorical features and reduced original dataset
