# ## 17 place solution

Competition: playground-series-s5e8
Rank: #17
Source: https://www.kaggle.com/c/playground-series-s5e8/writeups/17-place-solution

My solution is pretty straight forward with around 45 base models with CV score of 0.97734, which achieved private score of 0.97757. Most of the models are trained with similar hyperparameters, but different seeds and learning rate.

In the case of MLP, using of cyclical features as suggested by @yekenot (discussion [here](https://www.kaggle.com/competitions/playground-series-s5e8/discussion/596888)) gave nice boost to both CV and LB. The target encoding function in the work of @mahoganybuttstrings in this [notebook ](https://www.kaggle.com/code/mahoganybuttstrings/pg-s5e8-single-xgb-cv-0-975782-lb-0-97681)was very helpful.

Congratulations to all the winners and happy kaggling!!!
