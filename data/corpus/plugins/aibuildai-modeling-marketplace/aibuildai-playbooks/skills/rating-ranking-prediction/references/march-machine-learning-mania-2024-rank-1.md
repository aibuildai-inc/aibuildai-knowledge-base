# 1st Place Solution

Competition: march-machine-learning-mania-2024
Rank: #1
Source: https://www.kaggle.com/c/march-machine-learning-mania-2024/discussion/493793

**2024 March Madness Mania Competition**

[Notebook](https://www.kaggle.com/code/jaredcross/making-silver-based-predictions)

Private Leaderboard Score: 0.05313
Private Leaderboard Place: 1st

**Background:**

I am a science and statistics teacher, and make baseball player projections and do consulting work in baseball R & D.  Students in my high school Advanced Statistics class built decision trees to participate in [Titanic - Machine Learning from Disaster](https://www.kaggle.com/competitions/titanic) early in the school year and a number of them participated in this contest as well.  I'm planning to have them participate in [Spaceship Titanic](https://www.kaggle.com/competitions/spaceship-titanic)  in the coming weeks and they've also used a number of Kaggle Datasets for individual projects.  In short, I really appreciate what Kaggle has contributed to my classes.

**Solution:**

In the past, I've played around with [ELO ratings](https://www.kaggle.com/code/jaredcross/flexible-elo-code) and [mixed effects models](https://www.kaggle.com/code/jaredcross/log5-logistic-regression-and-bradley-te) to make predictions but this year I kept it simple.

My submission started with Nate Silver's ratings for Men's and Women's teams. It then gambles by making South Carolina's women's team and UConn's men's team super teams (good enough to win all of their games in every simulation).  In retrospect, this may have been more gambling than I should have used to maximize my chances of finishing in the top 8.  

A simple formula turns the differences between the ratings of two matched teams into the probability of a team winning (with a bump for home-field advantage for the top 4 seeds in the first two rounds of the women's tournament):

```r
msilver_wpct = function(pwr1, pwr2){
    pred_pt_margin = (pwr1-pwr2)
    tscore = pred_pt_margin/11
    pnorm(tscore)
}

wsilver_wpct = function(pwr1, pwr2, home=0){
    # home = 1 (home), 0 (neutral), -1 (away)
    hfa = 2.73*home
    tscore = (pwr1 - pwr2 + hfa)/11.5
    pnorm(tscore)
}
```


The tournaments are then simulated 5000 times which takes under an hour.  My thinking was that 5000 simulations was sufficient to make my average bracket quite close to the expected probabilities (according to the Silver ratings).


**Closing Thoughts:**

I enjoyed the format and liked the idea that we were submitting (a portfolio of) regular brackets.  I also like that the format has changed over the years because I think that makes us all write new code and devise new strategies and that's fun.  I do think that predicting point differentials evaluated by MAE ([as Jack Lichtenstein suggests](https://www.kaggle.com/competitions/march-machine-learning-mania-2024/discussion/492761)) has the potential to be a great format.  I'm looking forward to reading all of the solutions to this year's format!
