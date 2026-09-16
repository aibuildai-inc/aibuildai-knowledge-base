# 4th Place Solution

Competition: ncaaw-march-mania-2021
Rank: #4
Source: https://www.kaggle.com/c/ncaaw-march-mania-2021/discussion/230859

Hi Kagglers,

Here's the code for my solution: https://www.kaggle.com/aburkard/4th-place-solution

It's an OLS regression to predict the point spread using three features:

- Difference in scoring efficiency margin (weighing more recent games more highly)
- Difference in Draw Foul efficiency
- Difference in 3pt efficiency

where the efficiency ratings are [KenPom](https://kenpom.com/) style ratings that adjust for opponent and tempo. These can be calculated via linear regression where the features are a one hot encoded matrix of team ids. I experimented with a few different ratings and interactions between them, but these were the ones that gave the best score after leave on out cross validation.

I then fit a logistic regression, to the predicted point spreads as well as an interaction with the mean squared residuals calculated for scoring efficiency margin (unweighted). The intention here is that teams whose KenPom ratings are more predictable from game to game are more consistent, and so we'd want to predict a higher probability that the "better" team wins.
