# 54th place solution aka close but no silver cigar

Competition: mens-march-mania-2022
Rank: #54
Source: https://www.kaggle.com/c/mens-march-mania-2022/discussion/317077

Almost the same as my WMM solution https://www.kaggle.com/competitions/womens-march-mania-2022/discussion/316863 but with some different approach.

Different models to the ensemble:
https://www.kaggle.com/jonbown/creating-sequential-data-for-random-forest-in-r
https://www.kaggle.com/imoore/2019m-1st-solution-with-parameter-optimization
And 1 own Stacking regressor with 10 regressors models AdaB, Xgb,LGBM, CatB, RFR, LR, RidgeCV, GBR, HGBR and LSVR.

In contrast to WMM the "weighted moving target ensemble" described in the post above worked better than mean ensemble, a strategy with need of some luck as many other trix and fix I guess.

Final clipping:
finalsubmission['Pred'][finalsubmission['Pred'] <= 0.2] = 0.05
finalsubmission['Pred'][finalsubmission['Pred'] >= 0.9] = 0.95
