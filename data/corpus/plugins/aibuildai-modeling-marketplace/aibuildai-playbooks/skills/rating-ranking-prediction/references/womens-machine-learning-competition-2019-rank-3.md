# 3rd Place - Solution

Competition: womens-machine-learning-competition-2019
Rank: #3
Source: https://www.kaggle.com/c/womens-machine-learning-competition-2019/discussion/90156#latest-520707

Shout out to the organizers of this tournament. It was quite the rollercoaster ride as the tournament played out!

I used a linear regression model to predict the spread of each match-up. With pymc3's API, I was able to obtain team-level posterior distributions on FG attempts and their percentages. This allowed me to finally simulate a high number of match-ups between two teams and express a team's chance of winning as a function of the number of simulated matches won. Without a doubt, forcing high seed/low seed matches definitely helped with my final model score and this idea was inspired by @raddar from his 2018 winning solution.

If anybody cares, here's a breakdown of my approach (copy+pasted from my README.md):

&gt; A linear regression model was put together using pymc3. All data pre-processing and post-processing steps were handled using both numpy and pandas.

&gt; i. FG attempts as team-level Gammas vs Normals (or Poissons for counts) to allow using loc/scale parameters instead of mu/sd
ii. FG percentages as team-level Betas vs Bounded (0,1) Normals to allow using loc/scale parameters instead of mu/sd
iii. League-level Half-Cauchy error term to soak up some variance
iv. Assume team's scoring propensity is normally-distributed and treat as observed variable
v. Assume points spread between two teams is normally-distributed and treat as observed variable

&gt; Evaluation of odds of one team winning over another was done via simulation of team-level posterior scoring propensities (calculated from above).

&gt; i. Simulation for each match-up produces an output distribution representing the spread of each simulated match
ii. The win probability is determined by the number of simulated matches won over the total number of simulated matches
iii. 5000 simulations were run for each potential match-up
iv. Matches exempt from this process for overridding odds are high seed/low seed tournament matches

The Github link to the solution can be found below:
https://github.com/duythedewey/Kaggle_NCAA_Womens_2019
