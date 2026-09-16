# 9th Place Solution

Competition: march-machine-learning-mania-2023
Rank: #9
Source: https://www.kaggle.com/c/march-machine-learning-mania-2023/discussion/400151

So it's the time of the year again ... As someone who doesn't usually follow basketball, this is probably the time of the year where I become a basketball "fan". Although I must say that even as a casual observer, I was deeply engaged and really enjoyed watching some of the matches, especially towards the end. 

# Models

## Women's Bracket

I didn't do too well for the women's side of the competition with a Brier Score of about 0.160 mainly because my model was pretty simple but I felt like the predictions were not decisive enough even thougn I only fed it simple features.

I used Theo's Logistic Regression as a baseline and added some of my own features, namely the Aggregates (MIN / MAX / Mean / Median / etc ...) of important features like Point Difference / Total Points / etc ... then did some tuning. However, I believe Theo's original notebook did way better than my women's prediction.

Didn't spend too much time on the women's model because from prior iterations of the competition, I was rather certain that there wouldn't be too many upsets for the women's side and I was just focused on building a model that used the most (logically) "straightforward" features and did not dive into matchups at all.

## Men's Bracket

It was likely my men's bracket that pushed me up the LB. I had an overall Brier Score of about 0.195 and given how many upsets there were this year, I am pretty happy with the score. Similarly, from past iterations of March Madness, I think it is easy to come to a conclusion that the Men's side of the bracket will likely have more variance. I decided to build an XGB model with the following features :

- Rankings based on External Rating System (Sagarin, Pomeroy, Moore, 538 ratings, etc ...) just to name a few. 
    - Ranking changes throughout the season
- Win Rates & Point Differentials (Away / Home) 
- Team Box Scores (Aggregates)


Also, as I have yet to mention, there were lots of discussions on how Brier score could encourage more 1/0 flipping and Overriding gambling. As a risk averse person, this is not something I would do but got me thinking about the prediction distribution. 

I spent alot of time here, trying not to repeat my mistakes in last year's competition where my predictions were too conservative. Here are some of the more interesting things I tried:

- Different Forms of Scaling as Post processing (Didn't work)
- Added a `Risk Appetite` feature based on a variety of features (i.e Head-2-Head Matchup results, Seed differences, etc ...) to manage the prediction distribution
- Ensembling with a Simple Logistic Regression Model

Surprisingly, the last one worked the best (by far) and I was able to get a set of predictions I was happy with that were not too conservative and of course not too extreme.


**CV Stuff**

While I think CV is important for all ML problems, I didn't really look at it too much when I was trying to play around with the prediction distributions and added / removed features as long as there wasn't a drop in CV across ALL the seasons used as validation. My final ensemble actually had a lower Overall CV than just the XGB model itself.
