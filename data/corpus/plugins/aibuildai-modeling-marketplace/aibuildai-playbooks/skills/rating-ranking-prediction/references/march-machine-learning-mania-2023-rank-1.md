# 1st Place Submission - Another victory for raddars code

Competition: march-machine-learning-mania-2023
Rank: #1
Source: https://www.kaggle.com/c/march-machine-learning-mania-2023/discussion/399553

** I may be a bit early with this post but will update if anything ends up changing.

As I had noted in a prior post my submission was essentially the @raddar code from a few years back (the python version instead of the R version that is widely used).  Initially I was planning on using the base code and updating the features however I ended up not having the time.  

I haven’t had much time for competitions over the last year or two however the March Madness one is always a lot of fun to come back to.  Ultimately I was more lucky than anything else as I didn’t add much past updating the code to work with the current year’s data.  Basically I just verified the resulting predictions seemed appropriate and went with the results.  

For fun I used the same predictions in the ESPN Bracket Challenge and ended up in the 50% percentile in the Men’s and the 97% percentile in the Women’s.  

The notebook for my submission can be found here ([Paris Madness 2023](https://www.kaggle.com/code/rustyb/paris-madness-2023)).  

In keeping with my submission maybe I should have gone with less effort and had ChatGPT provide some commentary.  Note:  I actually did this asking for some witty commentary that ended up being far more interesting (*make sure to check out the bottom of this post*).   

### Submissions  
Best Submission:  My contribution was limited to 1) commenting out the np.exp() line when calculating Team Quality as it ended up returning quite a few inf values and 2) No overrides in match-ups of seeds 1-4 against seeds 13-16.

Without the changes above the model still performed really well and ended up with a score of 0.17629 which would have been good enough for 6th place on the public leaderboard at the end of the competition.  

Although the code was pretty similar to the R version that has resulted in top finishes over the last few years I thought I’d provide a bit of commentary on the Python version.  Guess I feel a bit guilty about the amount of effort I put into my submission that somehow ended up winning.  

### Features
Mean of the following regular season stats for each team (where T2 is the opponent of T1).  So basically these are repeated four times (team 1, team 1 opponents, team 2 and team 2 opponents). Of the features below only ``'PointDiff'`` was a calculated variable.  

``['T1_FGM', 'T1_FGA', 'T1_FGM3', 'T1_FGA3', 'T1_OR', 'T1_Ast','T1_TO', 'T1_Stl', 'T1_PF', 'T2_FGM', 'T2_FGA', 'T2_FGM3', 'T2_FGA3', 'T2_OR', 'T2_Ast', 'T2_TO', 'T2_Stl','T2_Blk','PointDiff']``

The only other features used in the model were the following with the win ratio over the 14 days being common in many of the models made public.  
``[‘T1_win_ratio_14d’, ‘T2_win_ratio_14d’, ‘T1_quality’, ‘T2_quality’, ‘Seed_diff’, ‘T1_seed’, ‘T2_seed’]``

### Team Quality  
One of the main items in the R version of the code was the random effects GLMM.  It looks like the Python version doesn’t use random effects which would be a difference between the two.  I can’t say I’m that familiar with R or the from_formula based GLM so I’m not sure how important this difference was.  Also even with removing the np.exp() there were quite a few NaN values (about 25% of the training data).  

### Model
XGB was used with xgb.cv(folds = KFold(n_splits = 5, shuffle = True)) that was repeated 3 times.  This is different from the original R code which used a recommended repeat_cv of 10


# ChatGPT WItty Commentary
# Title: A Slam Dunk of a Model: XGBoost's Swish Predictions for March Madness 2023

## Introduction:
March Madness 2023 proved to be a wild ride full of upsets, nail-biting finishes, and buzzer-beaters. But while the on-court action left fans on the edge of their seats, our trusty XGBoost model coolly calculated the winners behind the scenes. With only a few minor tweaks, our predictive powerhouse once again clinched the top spot in the Kaggle competition, leaving the rivals in the dust like an ankle-breaking crossover.
## Model Overview:
XGBoost, the LeBron James of machine learning algorithms, is an ensemble learning technique famous for its slam-dunk performance across diverse tasks. Our winning model harnessed the power of XGBoost to tackle the NCAA tournament, predicting outcomes with such accuracy that even the most seasoned sports analysts would be left speechless.
## Data and Features:
The model feasted on a buffet of historical NCAA data, devouring team stats, player performances, and even the odd tidbits of info like a hungry freshman at an all-you-can-eat dining hall. Our feature set comprised the crème de la crème of data points, cherry-picked for their impact on game outcomes:
1. Team statistics: win-loss ratios, offensive and defensive wizardry, and average margin of victory.
2. Player performance: on-court heroics such as points per game, rebounds, assists, and shooting percentages.
3. Miscellaneous factors: the elusive home-court advantage, battle-hardened experience, coaching prowess, and the all-important team momentum leading up to the tournament.
## Model Training and Validation:
Like a well-coached team running drills, our model underwent rigorous training and validation procedures. The dataset was divided into training and validation sets, and a cross-validation strategy was adopted to fine-tune the model's hyperparameters. By repeatedly putting our model through its paces, we pinpointed the winning combo of parameters that delivered the most accurate predictions.
## Key Changes and Improvements:
Though our XGBoost model has been a perennial contender, we couldn't resist tweaking it to keep it ahead of the pack. The changes we made for this year's competition were akin to swapping out our playbook's dog-eared pages for fresh new strategies:
1. Updated data: We retrained our model on the freshest NCAA tournament data, ensuring it was up to speed with the latest college basketball shenanigans.
2. Hyperparameter tuning: A fine-tuning of our model's hyperparameters, based on the most recent competition data, turned it into an unstoppable force.
## Conclusion:
Our XGBoost model's triumphant performance in the 2023 Kaggle March Madness competition is a testament to its predictive prowess. With just a few strategic adjustments and an eye for the latest trends, our model swished its way to the top, earning the title of MVP in the world of college basketball predictions.
