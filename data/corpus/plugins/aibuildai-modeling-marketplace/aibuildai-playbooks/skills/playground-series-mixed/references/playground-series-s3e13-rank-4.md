# #4 solution - simple model WITHOUT CV ;-)

Competition: playground-series-s3e13
Rank: #4
Source: https://www.kaggle.com/c/playground-series-s3e13/discussion/406812

Hey guys. Sorry for posting the code so late. Below is a summary of my approach.

The dataset was quite small if we consider the number of levels of the target variable, so I assumed that:
- CV with low number of folds is not a good idea
- GBT/NN + wrong validation = overfitting

What I did instead was: RandomForrest + OOB scores (to not split the data with so many levels) + Optuna.

The results (OOB, public LB, private LB) turned out to be perfectly correlated.

[Here is my full code.](https://www.kaggle.com/code/mateuszgrzybpl/4-solution-randomforrest-optuna-oob-score)
