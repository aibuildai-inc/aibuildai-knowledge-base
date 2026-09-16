# #3 Solution

Competition: predicting-red-hat-business-value
Rank: #3
Source: https://www.kaggle.com/c/predicting-red-hat-business-value/discussion/23803

First of all, I’d like to thank Loiso for the LB 0.987, group_1 and date trick kernel, and radar for the 0.98 xgboost on sparse matrix kernel as these sparked my interest into the unique challenges of this competition.  My model was just a simple xgboost, but it was the data where I discovered some insights that helped pushed me to 3rd place.  Here are my insights:

1. Expanded upon the group_1 and date trick.  I added more categories to Loiso’s script.  Because AUC only cares about the order of predictions, the actual numbers I assign are arbitrary as long as they are in the correct order.  I used numbers closer to 0 or 1 when I was more confident.

0/1 – Assign a perfect score only if the group_1 and date was the same between train and test.  Based on training data, it’s impossible for the outcome to vary within a group and date combination.

0.05/0.95 – Used when the train dates surrounding the test example where either both 1 or both 0.  Although rare, there is some slight chance that the outcome could change twice causing a wrong answer.

0.1/0.9 – Used when the test example is before or after all train examples.  This is less confident than 0.05/0.95 because only 1 outcome change is required for a wrong answer.

0.025/0.975 – Same as 0.05/0.95 except the outcome already changed twice elsewhere in the group.  It’s unlikely that an outcome changes more than twice.

0.075/0.925 – Same as 0.1/0.9 except the outcome already changed twice elsewhere in the group.  It’s unlikely that an outcome changes more than twice.

0.5 – Used when the train dates surrounding the test example do not agree.  These are the hardest to predict because the outcome could easily go either way.  Luckily, these aren’t very common.

0.497 – Used for the test groups that don’t exist in the training data.  The strange 0.497 number was left over from an old process for comparing predictions to the 0.5 group.

2. Originally I was weighting the above scores with an xgboost prediction, but it wasn’t until I switched the process around that I really started to get competitive submissions.  I mimicked what the leak would have looked like on the training data by looping thru every person in training data and finding what that person’s above score would be had they been in the testing data.  I then used this score as a predictor in my xgboost model.

3. char_10 is a tricky variable to use as one-hot encoding because it has so many levels.  I found the leave one out technique to be better, but there was risk of overfit.  I did a modified leave one out technique by leaving out a whole person and calculating the average outcome for the rest of the people in each char_10.  This ended up being a pretty good predictor.

4. I found group_1 to be a better predictor when left as continuous.  Low values of group 1 are more likely to be 1.

Group by Outcome.png

5. My model was at the activity level, but activities within the same group are likely to have the same outcome.  I found an adjustment after making my predictions to address this, and it helped improve my score.  I found the most extreme prediction (farthest from 0.5) within a group.  I then gave 90% weight to this prediction at the group level and 10% weight to the raw prediction at the activity level.

6. And yes, I did the rule based overrides to correct for ML model shortcomings that radar mentioned doing in this post: https://www.kaggle.com/c/predicting-red-hat-business-value/forums/t/23786/long-story-of-1-solution
