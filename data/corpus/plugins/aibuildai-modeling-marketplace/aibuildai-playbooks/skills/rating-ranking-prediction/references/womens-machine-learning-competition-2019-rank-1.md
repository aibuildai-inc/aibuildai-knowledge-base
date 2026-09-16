# 1st Place Solution (Be Aggressive, be be Aggressive)

Competition: womens-machine-learning-competition-2019
Rank: #1
Source: https://www.kaggle.com/c/womens-machine-learning-competition-2019/discussion/88451#latest-514486

Before I get into the explanation of how I made my predictions (sure to be a boring letdown), I would like to thank Jeff, Addison, and everyone at Kaggle that put work into administering this wonderful competition.

As an actuary, I work extensively with obtaining and analyzing large quantities of data on a regular basis. Furthermore, basketball is my favorite sport, both to play and to watch. Quantitative analysis of basketball, in particular college basketball, has been a passion of mine for the past decade. I have written multiple articles on the website I host (1-3-1 Sports) analyzing college basketball from a quantitative perspective. Additionally, I also act as a contributor to the website Busting Brackets, where I analyze basketball in a more qualitative manner.

I used simple linear regression in order to create a projected margin of victory for all possible iterations of games between each of the 64 teams playing in the tournament. The margin of victory was generated using offensive and defensive efficiency metrics calculated for each team using a points per possession methodology such that possessions are approximated as Field Goals Attempted – Offensive Rebounds + Turnovers + .5 * Free Throws Attempted. I took these calculated margins of victory and compared them to the power scores and probabilities produced here (https://projects.fivethirtyeight.com/2019-march-madness-predictions/womens/?ex_cid=rrpromo) as a reasonability check. Finally, I manually input predictions for as many match-ups as I could given the submission deadline. I prioritized prediction inputs in order from most likely games to be played (e.g. guaranteed-first round matchups) to least likely games to be played in order to optimize my available time.

My exposure to years of playing, coaching, and analyzing basketball allowed me to familiarize myself with the best indicators of victory within the sport. Time and time again, my research has shown that offensive and defensive efficiency work best and keep models simple.

Even with my exposure to the sport, I think more than anything what allowed me to achieve my final score was the aggressive predictions that I made. Take that with a grain of salt, because this methodology for making predictions will result in a wildly volatile performance year over year. When I finish dead last in next year’s contest because of an unexpected upset, don’t be surprised!

To provide a tangible example, I predicted all 1 through 3 seeds would win their first round games with 99% probability. This was based on the fact that in the past 20 years of the Women’s NCAA Tournament, one, two, and three seeds are undefeated in the first round against fourteen, fifteen, and sixteen seeds. Considering only one of these twelve games had a single digit margin of victory during this year’s tournament, in hindsight, it seems like a strategy of risk worth taking to have a marginal advantage against other scores.

In the coming months, my plan is to create a much more robust methodology for creating accurate probabilistic predictions, so that I can hopefully share that (and use it) in the 2020 edition of this competition.

I hope everyone had a great time during this competition, and I hope it continues to expand (both in participation and data availability) in years to come!
