# 22th solution

Competition: mlb-player-digital-engagement-forecasting
Rank: #22
Source: https://www.kaggle.com/c/mlb-player-digital-engagement-forecasting/discussion/261020

First of all, thank you very much to those who supported me, upvoted EDA, and excited me! I enjoyed experiencing various things.　The top ones are really amazing. I respect them.

I shared my solution in detail. (日本語解説ありです). 
This is the 12th solution before the train data is updated(leaked).

https://www.kaggle.com/chumajin/12th-before-train-update-solution-english


But I don't know the actual result. If I got worse score or submission error, please laugh me... ( I will laugh at my own)


[Short summary]
* I used only optuna and LGBM. It was Public LB1.3019 by ensemble what was created by changing the features.

* CV is 5 kfold by the average of from target1 to 4.

* In my 1st phase, I used the merged code that was published by kaggle staff. This LGBM model got 1.3490 score.

* In my 2nd phase, I added the features about the statics of target values more than 31 days ago because we know the correct answer. This model got better to 1.3373. I used GCP because of memory insufficient.

* In my 3rd phase, I used the log scale of target value and omitted 0 and 100 value. I fount the clean histogram if I use the log scale of target values. This model got better to 1.3256.

*  I ensemble models made with other features and models made for each position.
This models got better to 1.3144.

* Moreover, no hitter is very high targets value. So I correct it(1.3073). And I found Shohei Ohtani found that LGBM's predictions did not match, and corrected the difference(maybe this is overfit).
Finally I got to 1.3019.

Thank you so much, good luck for everyone !
