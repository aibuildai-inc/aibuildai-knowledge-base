# 5th Place Solution Outline

Competition: ncaaw-march-mania-2021
Rank: #5
Source: https://www.kaggle.com/c/ncaaw-march-mania-2021/discussion/231872

Congratulations Bojan for decisively winning this competition, and congrats to 
everyone who participated! This competition has been a wild ride for me and resulted in 
my first gold.

My own solution has some fairly novel components and only a
single game override. 


Offensive and defensive power ratings (OPR and DPR):
For calculating OPR and DRP we want to solve the equation:
Team 1 Score - Team 2 Score = OPR Team 1 - DPR Team 2

This can be solved to minimize MSE rather easily with just a linear regression.
However, that approach doesn't capture much about the consistency of the team's scoring 
abilities, so instead I solved this using different expectile losses with CVXPY. Then 
we can use the OPR and DPR values at different expectiles as features for the model.
I credit the introduction of expectile OPRs and DPRs as being the reason my 
NCAAW model substantially outperformed my NCAAM model.     

In addition, I calculated expectile and average metrics for the score difference
for each team (though probably unnecessary, I did this with a convex optimizer).

Model: I used an ensemble of 4 different LightGBM models to make probabilities
out of my OPR and DPR features. Since I created variations of these features
for most columns in the datasets, I used only a subset of only the strongest,
and generally directly score related, features for my final model.

Override: I hedged my bets for the game where my 4 models varied the most in the 
first round. This happened to be the SF Austin, Georgia Tech game, so I had one 
submission bet it all on SF Austin, and another one bet it all on GT. 

Some other notes: I used all the games data throughout history in and out of tournament for 
training the models, but considered each team in a different year as different. I also weighted
recent years and tournament games higher than the rest of the data.
