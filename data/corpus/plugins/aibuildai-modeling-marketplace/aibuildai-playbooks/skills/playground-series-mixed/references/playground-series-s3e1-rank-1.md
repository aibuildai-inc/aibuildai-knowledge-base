# That was a surprise! Here is the 1st place solution....

Competition: playground-series-s3e1
Rank: #1
Source: https://www.kaggle.com/c/playground-series-s3e1/discussion/377137

First - thanks for the fun competition, great public solutions and contributions! 👍

This will be a short summary but still as some maybe are interested in what FE/models etc. included to the solution.

Didn't had much time for this competition while doing other as well but joined the competitions for trying some new frameworks versions in this specific dataset.

I finally picked the AutoGluon framework and its tabular predictor for the task.
"AutoGluon-Tabular: Robust and Accurate AutoML for Structured Data" - https://arxiv.org/abs/2003.06505

For the FE part I used a public notebook https://www.kaggle.com/code/dmitryuarov/ps-s3e1-coordinates-key-to-victory ,  credit to author!

The AG trained solution is an weighted ensemble of many 8 fold trained common architectures as xgb,lgbm,catb,RF,NN etc and it also used bootstrap aggregation and stacking(3 levels for this one).

The final local CV score was 0.5006.

That's it! 🙂
