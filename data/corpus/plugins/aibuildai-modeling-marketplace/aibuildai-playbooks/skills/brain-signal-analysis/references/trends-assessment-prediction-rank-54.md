# 54th Private & 44th Public Place Solution

Competition: trends-assessment-prediction
Rank: #54
Source: https://www.kaggle.com/c/trends-assessment-prediction/discussion/162845

Congrats to all the kagglers who participated in this competition!

This one feels quite pleasant for me and my teammate since it has brought us to the Competitions Expert tier with a good ranking! :)

We've started out with RAPIDS-based blending of classical ML models inspired by [this notebook](https://www.kaggle.com/tunguz/rapids-ensemble-for-trends-neuroimaging).
We were trying to pick coefficients via [SciPy linear least-squares bounded problem solver](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.lsq_linear.html), but we have faced the overfitting bound soon enough to give up this technique.

The game-changer for us was [Multi-Layer model](https://www.kaggle.com/david1013/trends-multi-layer-model) generously shared by @david1013. We started with blending 60 output .csv's with different seeds. An important thing is that our validation has shown that this ensemble underestimates most of the values, especially **age**. So we tried out an element-wise maximum of 60 predictions, and it was much better w.r.t. both the CV and public LB.

However, **age** value remained lower than expected. Running out of time, we tried to pick a good multiplier for it as a part of post-processing. The best we have tried was **1.005**, so we've stopped on this predictor.

Thanks!
