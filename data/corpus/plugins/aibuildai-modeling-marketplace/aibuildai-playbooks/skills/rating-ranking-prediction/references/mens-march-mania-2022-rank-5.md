# #5 Solution

Competition: mens-march-mania-2022
Rank: #5
Source: https://www.kaggle.com/c/mens-march-mania-2022/discussion/317566

Firstly, I'ld like  to appreciate the competition hosts, Kaggle team ( @addisonhoward ), @jeffsonas  for the organization of this competition.
I would also like to congratulate @amirghazi for winning and encourage those that lost top spots or medals by one or two places or games

About me:
its my first time participating in this competition & must say its been really fun. its my first time building basketball prediction models also. i have had some experience in the past building models for european football home/away team total [full-time](https://www.kaggle.com/code/tomy4reel/jarvis)/[half-time](https://www.kaggle.com/code/tomy4reel/jarvis-ht) goals prediction using linear regression/other regression models & a variety of data/meta-data across seasons (it's usually quite easy to overfit & simpler models tend to perform better over time/seasons)
Anyways, i took @zachmayer [course on datacamp](https://www.datacamp.com/courses/advanced-deep-learning-with-keras-in-python) where i got basic idea and motivation for this competition.

Public Solution:
my #5 place solution python code can be found [here](https://www.kaggle.com/code/tomy4reel/5th-place-solution-men-s-competition)

Datasets and Features:
i used only some of those provided by the competition hosts.
i used the current seeds, historical win stats, number of matches played in previous seasons which can indicate how far teams went

Model selection:
catboostclassifier with logloss objective and a few categorical features

cross validation:
was done with last 2 available seasons only; i didn't make any submission on the initial leaderboards 

What i didn't try: 
1. i didn't clip or modify my predicted probabilities
2. i didn't use any of @raddar model or dataset (because i didn't see them on time)

Model performance:
i didn't aim to win gold. i only tried to build a good model & mostly judged my performance initially by comparing my leaderboard positions with that of @raddar and @zachmayer; later on with @koki25ando , @fritzcremer @dynamic24 

Limitations:
1. Time: I started working on the competition late
2. knowledge: i had little understanding of some datasets and features. but the data description provided by hosts were very helpful. keep it up.👍
