# 7th Place Solution

Competition: march-machine-learning-mania-2023
Rank: #7
Source: https://www.kaggle.com/c/march-machine-learning-mania-2023/discussion/400116

# 2023 March Madness Mania Competition

[GITHUB Link](https://github.com/JackLich10/kaggle_2023)

## [7th Place Submission](https://www.kaggle.com/competitions/march-machine-learning-mania-2023/leaderboard)

```
Private Leaderboard Score: 0.17704
Private Leaderboard Place: 7th
```

### Background

I am a 2022 graduate of Duke University with a degree in Data Science. I currently work for a sports analytics company in Washington, D.C.. This is my second year entering this type of competition. I entered because I think it's super fun to forecast the results of such a crazy basketball tournament. Last year, I finished 100th in the men's only competition. I did not participate in the women's only competition last year. My men's model was written 2 years ago for fun while my women's model I created this year in ~4 days to enter this competition.

### Methodology

All coding was done in R. The basic methodology behind both of my models is to:

1) generate team ratings via ridge [(`glmnet`)](https://glmnet.stanford.edu/index.html)/mixed effects [(`lme4`)](https://github.com/lme4/lme4) modeling predicting point difference, offense/defense efficiency, offense/defense pace, etc. (i.e. what is team A's impact on offense efficiency, score difference, etc. after controlling for opponent, home court, etc.?)

2) use team ratings from 1) to predict game level score difference (via [XGBoost](https://xgboost.readthedocs.io/en/stable/))

The most important features to predict game level score difference are the various team ratings from 1). Each of these models are trained after each day of each season. As such, it takes multiple days for the entirety of this modeling framework to run for the first time. A simple GLM converts predicted score difference to predicted win probability.

### Code

This [repository](https://github.com/JackLich10/kaggle_2023) has my men's model and women's model in their respective folders, `mens/` and `womens/`. Within each of `mens/` and `womens/`, the `run_all.sh` shell script specifies the order in which the files need to be run for the entire modeling framework to work. The `kaggle_preds.R` script combines the predictions into the final submission file.

The first time running everything, the scripts inside the subfolder `models/` within each `mens/` and `womens/` must be run to tune the hyper-parameters of the XGBoost models found within each workflow as well as to train the men's shot probability model (This tuning is done via the [`tidymodels`](https://www.tidymodels.org/) framework and using crossfold validation). You will also have to create `data/` folders within each `mens/` and `womens/` directories.

*NOTE*: My men's model uses outside public data via my own [gamezoneR](https://jacklich10.github.io/gamezoneR/index.html) R package. My women's model only uses provided data via the competition website.

### Manual Overrides

In one of my submissions, I manually set Oral Roberts win probability over Duke as 80%... that did not work out. In the other, I manually set Michigan State's win probability over USC as 80%... that one worked out. These overrides were dumb because I should have put them at 99% and I could have been in a very bad spot if both of them failed. For the women's bracket, I over-rid all 1 over 16 seeds at 99%.
