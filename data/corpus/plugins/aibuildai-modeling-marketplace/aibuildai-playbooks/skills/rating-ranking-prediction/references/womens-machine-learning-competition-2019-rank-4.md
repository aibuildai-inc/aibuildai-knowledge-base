# 4th place solution: 538 and Baylor Winning

Competition: womens-machine-learning-competition-2019
Rank: #4
Source: https://www.kaggle.com/c/womens-machine-learning-competition-2019/discussion/88462#latest-512204

First, thanks to the organizers and to Kaggle... and to 538 and their predictions and ELO ratings... and to Baylor!

My hope was to make predictions that were good enough and then add just enough guessing. 

Here's what I did in more detail:

1. Used 538's predicted probabilities for all first round games.

2. Used 538's ELO ratings to generate probabilties for all other games.

3. Adjusted second round games for home field advantage (the first round probabilities already included 538's home field advantage adjustment).

4. Gave Baylor 100% to win all games.

5. Created one bracket where the winner of the Chicago region beats the Albany region winner (100%) and another where Albany winner beats the Chicago winner (100%).

My code is here:
https://github.com/jfcross4/kaggle_NCAA/blob/master/stage2_womens.R
