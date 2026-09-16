# 31st place solution

Competition: home-credit-default-risk
Rank: #31
Source: https://www.kaggle.com/c/home-credit-default-risk/discussion/66010

Hi guys!

It is a little bit late but I want to share my simple solution. My 31st place is a best result for me. It is just 7 places after gold zone.

My model is a simple stack of 4 models:

- open solution with seed 0 from https://github.com/neptune-ml/open-solution-home-credit
- open solution with seed 90210 from https://github.com/neptune-ml/open-solution-home-credit
- kaggle kernel https://www.kaggle.com/aantonova/797-lgbm-and-bayesian-optimization
- my own XGBOOST model on mean / min / max / delta features with CV 0.796

The weights of models found by regression were: 0.23 / 0.12 / 0.3 / 0.35

Also, I added a little data leak that was found by @raddar https://www.kaggle.com/raddar/a-competition-without-a-leak-or-is-it

For stacking I used logistic regression and it gave me  Public LB 0.80651 / Private LB 0.80047 / Local CV 0.801.

I didnt spend too much time on competition and gold medal would not have been well-deserved for me.

So, silver medal and top 1% is a good result for me. Thank you all for great ideas in your kernels. I learned a lot. Happy new kaggling!

https://github.com/paveltr/home_credit_default_risk
