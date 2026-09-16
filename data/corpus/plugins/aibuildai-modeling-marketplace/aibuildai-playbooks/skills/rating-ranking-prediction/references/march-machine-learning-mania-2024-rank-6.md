# 6th Place Solution for the March Machine Learning Mania 2024 Competition

Competition: march-machine-learning-mania-2024
Rank: #6
Source: https://www.kaggle.com/c/march-machine-learning-mania-2024/discussion/497514

## **As usual, UConn shines during March Madness!!!**

## 6th Place Solution

## 2024 March Madness Mania Competition

## **Private Leaderboard Score: 0.05564**

## **Private Leaderboard Place: 6th**

I will start by congratulating all participants and thanking Kaggle and the organizing committee for this really nice contest year after year!

## **Background**

I am currently a fourth-year PhD student specializing in operations research at the School of Business of the University of Connecticut (UConn). Before starting my doctoral studies, I earned a Master of Science in Mathematics from Polytechnique Montréal and a Bachelor of Science in Actuarial Science from Université du Québec à Montréal (UQAM).

My graduate studies research has focused o on machine learning and optimization in the context of sports betting. Specifically, I have been working for several years on selecting multiple entries in a March Madness contest. My collaborators and I have developed several heuristics and tested our best heuristic on the 2023 March Madness contest organized by DraftKings.   The heuristics developed in our work aim to find the optimal collection of entries that maximizes the expected score of the maximum-scoring entry. We then show that the heuristic achieves better results than some of the best sports bettors in the world and has a 2.2% chance of winning $1 million. This paper is under revision (ArXiv to come).

Although I have been working on March Madness for several years, my focus hasn't been on predicting it. However, my expertise has allowed me to test and explore different prediction models. 

## **Methodology**

My implementation was done in Python. Over the years, my analysis has suggested that Nate Silver's ratings were typically robust enough and provided great insights into the outcomes of the March Madness tournament. The main thing missing from these ratings was some small adjustments for location. Due to time constraints, I entered the contest using Nate Silver's prediction and made the following small adjustments:
    
* For all teams in the men's tournament, I looked to see if they would be playing home, away, or neutral. I arbitrarily decided on that status depending on how far the game was being played from the university's campus (e.g., Storrs South-East (Brooklyn) and Storrs North (Boston) were both home games for UConn.)
  
   
In an ideal world, I would have also analyzed which teams played at home for the women's tournament as a deeper analysis of the injury list for each team. Lo Locations is known to have a major impact (especially in college sports), and injuries are also a big part of March Madness as every team does not rely on a very deep bench, emphasizing the importance of considering injuries before generating predictions. Looking back at my submission, I got lucky because I intended to reduce the rating of the UConn women's team considerably due to the restricted lineup.

Using both men's and women's ratings, I generated the 64 by 64 matrices using the classic Nate Silver ratings. Given the matchup of team A and team B with their respective ratings rA and rB, the probability that team A wins is given by P_{A, B}=max(1/(1+10**(-30.464*(rA-rB)/400))+homecourtAdvantage, 1) where homecourt Advantage is a +5% probability if only one of team A or team B plays at home. Although these small changes are rather simple, they were enough to allow me to finish in 6th position!

Given the binary aspect of such a metric, I think it is important to consider the uncertainty within the predictions itself. I was working on a 2nd submission, but I came up short on time (github: **githublink**). 
Some of the models previously tested were a seed-based model using logistic regression, the forecasting function provided by the R package cbbdata, and a boosted model developed using the location of the game, offensive ratings, and defensive ratings, as well as season-long and rolling window variables on team-wide statistics. 

## **Thoughts on the contest**

Although I have been working on this for several years, this was my first time submitting predictions. This year's process seems to be improving a lot compared to previous years. 

I may be wrong, but the metric may still penalize great participants suffering from a large upset. Taking this idea from Daily Fantasy Sports (DFS), allowing participants to submit more than two entries (e.g., 5 entries) would be interesting. Due to the high variance in the results of March Madness itself, I believe that the stochasticity within each parameter estimation is large, and two entries may not be enough to model that in the prediction. A strategy used by many participants seems to adjust the prediction matrix by manually fixing some of the games for their 2nd entry, but allowing participants to have more prediction matrices may help the better contestants outperform participants who fix some of the outcomes or who use a simple approach like this submission.

## Notebook submission:

Inputs
* Nate Silver's men's ratings: men2024.csv
* Adjusted Nate Silver's women's ratings: women2024
* Home game matrix for all matchups: MenHomegame2024.csv
    
Outputs:
* Men's team by team (pteam) probability matrix: men_pteam_2024.csv
* Women team by team (pteam) probability matrix: women_pteam_2024.csv
* Men's team by round (pround) probability matrix: men_pround_2024.csv
* Women's team by round (pround) probability matrix: women_pround_2024.csv
* Submission File: submission.csv


The code takes about 1 minute to compile everything. My code is available on my Kaggle.

Till next year!
Jeff
