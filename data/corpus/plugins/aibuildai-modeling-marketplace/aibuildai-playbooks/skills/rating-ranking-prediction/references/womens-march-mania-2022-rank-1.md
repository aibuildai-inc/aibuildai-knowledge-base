# 1st place solution/approach

Competition: womens-march-mania-2022
Rank: #1
Source: https://www.kaggle.com/c/womens-march-mania-2022/discussion/317817

First of all, I'd like to thank kaggle admins for hosting this year's competition. This is my second time participation into NCAA Basketball competitions hosted on kaggle. As well as last year's competition, it was quite enjoyable to watch games and check how my models work every day. I'd like to thank kaggle admins for organizing this competition.

In this repository/forum, I'd like to share my approach to this year's competition. However this type of competition is heavily influenced by luck, so my approach would not work next year by high chance. But, I hope some of you find my approach useful.

Basically, I submitted two types of models. One of them is "passive" model, the other is "aggressive/gambling" model. In short, my aggressive/gambling model ended up being the most accurate model throughout the competition (final leaderboard score was 0.35438). Although my passive model didn't work as well as the other one, still brought me better result than I expected. (In fact the final leaderboard score was 0.43574, bronze medal.)

In "passive" model, I submitted predictions calculated using only LightGBM without any post-processing. In "aggressive/gambling" model, I did override predictions of three teams.

I shared my R-scripts and Python notebook in [my GitHub repository](https://github.com/koki25ando/March-Machine-Learning-Mania-2022---Women-s).
Please let me know if you have any questions.

# Data Preparation

I wrote R scripts to set up datasets. 
I did no tricky things at all, I setup very basic features such as Seed info, teamid, team information...
For further complete information, please read my [R codes](https://github.com/koki25ando/March-Machine-Learning-Mania-2022---Women-s/tree/master/R_code) which exist in 'R' folder in this repo.

# Feature Engineering

To improve my cross validation score, I did very basic categorical feature engineering, such as count encoding, label encoding, target encoding...
See [this code](https://github.com/koki25ando/March-Machine-Learning-Mania-2022---Women-s/blob/master/py/LightGBM_stage2.ipynb) to check what's being done. 

# Modeling

I used only one model, LightGBM. That's it. I didn't even stacked any other models in the end.
I tried to make it as simple as possible. So I decided to use only LightGBM.
See [this notebook](https://github.com/koki25ando/March-Machine-Learning-Mania-2022---Women-s/blob/master/py/LightGBM_stage2.ipynb) for more information.

# Post-Processing

Here comes aggressive/gambling part. Basically I did override 3 teams prediction as every other participants in this competitions do.
I've read some articles, and chose three teams, "Stanford", "UConn", "South Carolina".

・[Full list of odds to win the 2022 NCAA Women’s Basketball Tournament](https://dknation.draftkings.com/2022/3/14/22976763/ncaa-womens-basketball-tournament-odds-2022-march-madness-national-championship-uconn-memphis)

In the end, three of them survived until FINAL FOUR. I guess I was just this year's lucky boy.
