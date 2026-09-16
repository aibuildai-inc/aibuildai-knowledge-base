# 1st place solution-Summary

Competition: lish-moa
Rank: #1
Source: https://www.kaggle.com/c/lish-moa/discussion/200736

This competition was full of learning for me, I had really great experience with my teammates @markpeng @kibuna  and @poteman. 
We didn't want to let you guys wait much for our approach, so I'm posting an overview of our final blend submissions during the final days. Mark will post the final original solution once we are done with recollecting all data and scripts. 
So, as mentioned by him earlier we chose two submissions based on the best Cv score and best leaderboard score. 
For maximizing cv score, @markpeng used optuna search and some other sklearn libraries, while for LB, I preferred choosing weights based on models correlation and leaderboard scores.
 


It seems like our cv - lb correlation was good, isn't much difference between the two submissions.
These were based on the blend of 7 of our best scoring models, all with high diversity.

**Best lb was based on the following models:**


**While Best cv was  based on these models:**

There isn't much difference in the models except for using 10 folds model in optuna search for maximizing cv score w/o using one additional model which was based on @cdeotte split.

**Score with / without CNN based models:**


Because of high diversity and low correlation, Mark's CNN based models were surely a great addition to our final score.
One of our main concerns about the competition was that we didn't use a drug_id split for our models, mainly the reason was lack of time and submission in the final days. So, we wanted to have a common cv split for all our models to get a better idea about improvement in cv scores. Decided to stick with the old cv scheme. Although, it did perform pretty well on the private leaderboard.
