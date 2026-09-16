# 24th place solution | Logistic model on top of Ensemble (XGB + Catboost + LGBM)

Competition: tabular-playground-series-sep-2021
Rank: #24
Source: https://www.kaggle.com/c/tabular-playground-series-sep-2021/discussion/276038

Hello Kagglers,

Pleasantly surprised to see my submission jump up by almost 80 places and end up in 24th place (I had almost given up after my submissions could not get me in the top 100 in the public leader board)
I would like to thank @mlanhenke, whose public notebooks helped me a lot to learn about Optuna and meta-learner.
 
My work involved stacked meta-learner or ensemble based on xgb, catboost and lgbm.

The approach was similar to what @mlanhenke did in his public notebook.

I just train different base models and save the oof_predictions to build a meta-set in the end. I also increased the stratified K-fold split to 10 to achieve better validation and more robust model.

As my final (meta) learner, I simple utilized a Logistic-Regression-Model to predict the probability.

Thanks and Regards,
Old Monk
