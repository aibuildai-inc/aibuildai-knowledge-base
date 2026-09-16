# 3rd Place Solution - Madtown Machine Learning Madness

Competition: mens-machine-learning-competition-2019
Rank: #3
Source: https://www.kaggle.com/c/mens-machine-learning-competition-2019/discussion/90254#latest-520985

# Team: Madtown Machine Learning Madness
- Jonathan (Jack) Smith - jgsmith725@gmail.com
- Jay Jojo Cheng - jay.jojo.cheng@gmail.com
- Young Lee - leey634@gmail.com

# Team Background
We are graduate students at University of Wisconsin - Madison.
- Jack works as a software engineer for Epic. He is currently pursuing a masters degree in computer science.
- Jojo is a first year PhD student in the department of Biostatistics and Medical Informatics. He spent a few years working in healthcare prior. He studied math in undergrad.
- Young will be graduating in May with masters in statistics and computer science. He interned at Amazon as data scientist and is currently searching for jobs in New York City.

# Summary
We used a logistic regression model to predict the winning probabilities. This worked well when one team was clearly better than the other.

However, in March Madness, there are many evenly matched games, which we assumed to be unpredictable. For these games, we randomly chose a team to win. Specifically, we chose the team with higher ID to win in our first submission, and the team with lower ID to win in our second submission. With this approach we were able to reduce our logloss from about 0.473 (top 200 in the leaderboard) to our final score of 0.427.

## Model
Our final model has only two features: 
1. TeamID
2. Points each team won by during regular season

Our model extends the [Bradley-Terry logistic regression model](https://youhoo0521.github.io/kaggle-march-madness-men-2019/models/bradley_terry.html). We estimate parameters that represent the level of every team based on the winning outcomes of regular season games. During training, we shared these parameters with another linear regression model that models the difference in the points earned by each team in a game as the response variable. By doing this, we were able to use the game scores, in addition to the winning outcomes, to estimate the team levels.

In this model, we deliberately limited the use of game-level features such as boxscore statistics. The main reason is that, for many standard models, these features would provide misleading information. To illustrate this point, suppose that Wisconsin scored 100 points and had excellent offensive efficiency rating in a game. Against a strong defensive opponent, this could mean that Wisconsin has very strong offense. Against a very weak defensive opponent, it’s plausible that Wisconsin’s offense is mediocre. Without knowing the opponent, game-level features provide incomplete information about the team.

We experimented with various combinations of boxscore feature engineering, ML models, and ensembling. In the end, our simple logistic regression was the best predictor.


## Random Guessing (Submission Strategy)

Some Kagglers used two submission to flip one game (100% to 0%). With our strategy described below, we flipped 31 closely-matched games (64% to 36%).

### Motivation

We noticed that logistic regression model correctly estimated its own uncertainty; when it predicts that a team will win with 50% probability, it is correct about 50% of the time. However, it’s impossible to get competitive log-loss by making many conservative predictions near 50% probability.  We hypothesized that when logistic regression is not confident, it's because the game is evenly matched and is unpredictable.

### Strategy
  
When the game is unpredictable (model prediction is less than 77%), we picked the team with lower ID to win with 64% probability. In our second submission, we predicted the other team to win with 64% probability. Assuming that team IDs have nothing to do with winning, this is just random guessing.

### Why does this work?
  
Under log-loss scoring, this strategy works if there is an uneven split between the number of correct and incorrect predictions. The more extreme the split, the better this works. For example, 70:30 split is better than 60:40. If these games are truly unpredictable, there’s about 30% probability that the split will be favorable for our strategy (binomial distribution). Luckily, this was the case for this year's tournament!

# References
- Model summary document as per the guideline: https://drive.google.com/open?id=1SR4OVaLd-qBUGBZD3Ml5W4bGpdeCZ9NVk8Tcv4XzqU0
- Minimal code to generate submissions: https://github.com/YouHoo0521/kaggle-madtown-machine-learning-madness-2019
- Our development repo contains code and notebooks we used to explore feature engineering and modeling: https://github.com/YouHoo0521/kaggle-march-madness-men-2019
