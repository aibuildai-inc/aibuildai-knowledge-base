# 2nd Place Model

Competition: mens-machine-learning-competition-2019
Rank: #2
Source: https://www.kaggle.com/c/mens-machine-learning-competition-2019/discussion/88805#latest-512670

Thank you to the competition organizers and sponsors, this has been a fun project for me over the years to learn how to code in R (hence why this code is fairly ugly because i didn't re-write when i learned more efficient ways of doing things).  The model itself is actually fairly simple in it's construction but fairly strong in it's predictions and it's love for Virginia.  I made no adjustments to the model and only made one submission (because I forgot about doing it until the night before) so I didn't utilize any game theory, which would have been smart.

It uses 8 basic variables to train and Score the model:
Each variable is the difference between the 2 teams playing in each category.
KenPom Variables:
Adjusted Offensive Efficiency, Adjusted Defensive Efficiency and Strength of Schedule Rating.
     These variables were pulled from the website and do have a slight bias as tournament data is incorporated in them for historical years so the fit is a little biased but it's readily available and easy enough to pull.

Final Massey Ratings, tested several of the rankings out and these seemed to work the best.

Calculated fields from the Data provided:
Season Defensive Rebounding 
Odd Features, last 30 days of the season:
Difference in the variance of game to game free throw percentage.
Difference in the variance of turnovers in the game to game free throw percentage.

Seed Difference: This variable seems silly to include but should account a little bit for the "eye test" used in seeding teams in things that aren't accounted for in other metrics.  It obviously biases the predictions towards the higher seeds, so i'd like to give a shout out to the committee for their great job of seeding this year.

Here is the link to my code on GitHub.
https://github.com/gjwierz/NCAA_Kaggle_2019
