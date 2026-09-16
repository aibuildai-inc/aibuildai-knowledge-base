# 40th place solution

Competition: womens-march-mania-2022
Rank: #40
Source: https://www.kaggle.com/c/womens-march-mania-2022/discussion/316863

First, congrats to the new champions South Carolina!

As summary this was ensemble of 3 models, R and Python, different features and with mostly LGBM and XGB.
Like many others I used solutions from top models from previous NCAA competitions with some minor changes. All credit to them.
https://www.kaggle.com/imoore/2019m-1st-solution-with-parameter-optimization
https://www.kaggle.com/code/svyatoslavsokolov/2021-ncaaw-first-step-v-3-stage-2-visualization

And 1 own model, a Stacking regressor with 10 regressors models AdaB, Xgb,LGBM, CatB, RFR, LR, RidgeCV, GBR, HGBR and LSVR.

All models were validated and tuned against this dataset and previous competitions.

This year I did skip the 0-1 clipping 😉 but still used high edges clipping in one submission and less in the other one.

In one submission I used weighted moving target, shifting predictions based on the model’s outcome, from worst to the best validation, hopefully towards the direction of the better predicted value. This went well last year and in the men’s competition this year, but an ensemble is also in place for the second submission, as one must rank the model’s performance in advance to work well, done by validation on prev. years outcomes.

So what to learn from this competition? That’s to come reading from the top solutions but from my point of view I will boost the middle range probabilities as well in one submission not just the edges, while still leave one sub. more or less untouched as previous year, hoping that the A.I. brain power beats the manually changes 😊
