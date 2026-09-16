# 4th Place Solution for the Regression with an Abalone Dataset

Competition: playground-series-s4e4
Rank: #4
Source: https://www.kaggle.com/c/playground-series-s4e4/discussion/499341

My approach was pretty simple. So, I will keep it short. All I did was to improvise this [notebook](https://www.kaggle.com/code/mustafakeser4/ps-s4e4-autogluon):

- OpenFE
- Target log transformation
- AutoGluon (with custom RMSLE metric and only tree-based models)
- Average of this [notebook's](https://www.kaggle.com/code/trupologhelper/ps4e4-lightgbm-only) output and mine (yeah, it's 50-50)

That's about it. To be honest, I didn't expect it to perform this well.  From what I have observed so far, AutoGluon performs really well on the datasets that has at least 50-60k observations.
