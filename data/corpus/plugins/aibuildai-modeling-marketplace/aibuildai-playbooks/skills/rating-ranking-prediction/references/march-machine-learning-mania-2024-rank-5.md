# 5th Place Submission

Competition: march-machine-learning-mania-2024
Rank: #5
Source: https://www.kaggle.com/c/march-machine-learning-mania-2024/discussion/497162

background
=========

Hi, my name is Stefan. I am an 18 year old high-school student from Brooklyn.
I have some background in ML (I took a year-long class and I have been studying independently since then) but I found little of my ML experience useful in this competition. 

 submission results
==============

- Leaderboard score: 0.05563
- Leaderboard place: #5

[submission notebook](https://www.kaggle.com/code/stefanwb/march-machine-learning-mania-2024) 

submission info
===========

I made initial predictions using Ken Rom's AdjEM data for the men's teams and Massey's power ratings for the women's teams.
 I made my calculations expecting 70 possessions, and assumed a normal distribution in order to calculate probabilities.

In my successful submission, I used a standard error value of 9 for the men's results, and 12 for the women's results. 

calculating t score to predict distribution (scale AdjEM to 70 possessions): 
$$\frac{0.7 \times (AdjEM_1 - AdjEM_2)}{\sigma_{M}}$$

For women's data, I used Massey's ratings and the same logic that I used for the Men's ratings (using Pwr in place of AdjEM, and taking account for hfa)
$$\frac{0.7 \times (Pwr_1 - Pwr_2 + hfa)}{\sigma_{M}}$$

For my successful submission, I only simulated 50 brackets and made no manual overrides to any predictions.

It's interesting to note that submitting more brackets makes my submission perform worse. My assumption is that by simulating only 50 brackets in this submission, the error in my prediction happened to more accurately match the error in actual results— the results of my simulated brackets seemed to deviate from the expected results in the same way that the actual games did. This was the strategy for my backup submission, and I'm surprised it worked as well as it did. 

Special thanks to KenPom and Massey for the ratings, @jaredcross for his [simulation code](https://www.kaggle.com/code/jaredcross/making-a-chalk-bracket-in-r), and Kaggle for hosting this great competition!
