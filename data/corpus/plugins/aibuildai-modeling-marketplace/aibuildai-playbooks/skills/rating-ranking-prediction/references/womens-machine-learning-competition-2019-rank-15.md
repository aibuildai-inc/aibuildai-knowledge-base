# 15-th place solution (a bit risky)

Competition: womens-machine-learning-competition-2019
Rank: #15
Source: https://www.kaggle.com/c/womens-machine-learning-competition-2019/discussion/88421#latest-524134

Without override: 0.36626
With override: 0.34766

Baseline: @raddar 's MAE optimization, transcribed in python with quality features generated from R code

A Few additions:

1. I tested out xgb, lgb and carboost separately. Since catboost's python custom metric is too slow and adding the cauchy objective in c++ and recompiling ended up not working, I ended up using catboost with rmse metric.

2. Baseline's 'quality' feature ended up being really useful, but my separate observation is: although some teams are equally competitive, one may be more stable than the other, so I engineered features such as truncated mean and standard deviation of quality.

3. Fitting a spline to the score-diff -&gt; win-prob curve will cause the graph to have two dips around prob = 0 and prob = 1, which is counter-intuitive: an increase in score-diff should only increase the winning probabilities. So I changed UnivariateSpline to IsotonicRegression.

4. Clipping final [0.025, 0975]

Overrides:

1. According to the bracket generation program I've found out that Baylor's winning probability is around 0.68, so I made a gamble to set Baylor's winning probability against the two most likely opponents in the final to 1, and it paid of pretty well!

2. Seed overrides as described in baseline. I used the same approach in Men's competition and get burned for that :(

Now looking back, I should've set Baylor's winning probability against all other teams to 1, which would be more risky than my current setup but will pay off better, and I think that's my solution's difference from a gold solution...... (more faith!) And it's kinda sad that I'm only 4 place away from a gold medal (and Grandmaster status requires a solo gold), so I guess I'll have to wait for another competition :)

PS: If there's enough interest I will upload my solution via Kaggle kernel.
