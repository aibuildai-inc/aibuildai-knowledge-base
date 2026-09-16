# #6 place - A quick reflection

Competition: playground-series-s4e8
Rank: #6
Source: https://www.kaggle.com/c/playground-series-s4e8/discussion/531330

First, huge congratulations to @optimistix who has been flirting with the top spot all summer, and finally got there. Also the only person in at least top 100 who didn't move at all from the public LB. How cool is that?

My solutions tend to be boring because they all boil down to the same thing: huge ensembles. This one was no exception but I didn't get to 40-50 models as in previous competitions - only to 25. In some order, they were:

| Type | # of models |
| --- | --- |
| LAMA TabularNN | 8 |
| AutoGluon | 6 |
| CatBoost | 4 |
| Keras FM | 3 |
| xLearn FM | 2 |
| LightGBM | 1 |
| XGBoost | 1 |

Beyond that, nothing fancy. Picked a model that had best CV and that ended up being my second best model overall.

The best solution I had was actually by hill climbing, which picked only 13 of the above-mentioned 25 models. Yet it had quite a bit lower CV score, so there was no reason to pick it. It has the same 5-decimal score as the model I picked.

My best individual models were by AutoGluon, but those shouldn't be counted because they are ensembles. From actual single models, four Keras factorization machines were the best (private scores 0.98413-0.98433). They treat all the variables as categoricals and model their interactions. After that the best 8 models were still LAMA TabularNNs, followed by xLearn factorization machines and CatBoost models.

**It would appear that modeling all (or most) variables as categoricals was a way to go.**
