# #2nd Place Solution(Team Peaky Blenders): Blends Of Blends.

Competition: playground-series-s4e5
Rank: #2
Source: https://www.kaggle.com/c/playground-series-s4e5/discussion/509410

Thank you to Kaggle for organizing tabular playground competitions. I've learned a great deal from participating in these competitions. I also gained valuable insights and modeling from public notebooks and discussions. Thank you to everyone for sharing so much knowledge.

Thank you @mdoroch for teaming up , I had wonderful time working with you and your contribution were incredible for our team😀!

**Our Solution:**
- For feature engineering,
- Features suggested by @ambrosm and @siukeitin were used and features from top notebook
[Team Oxygen: AutoML 2nd Place](https://www.kaggle.com/code/nischaydnk/team-oxygen-automl-2nd-place)

- Our solution contained of total of **56(excluding autogluon)**.These 56 models contained models hyperparamter from public notebooks to be specific,
(hyperparamters from these notebooks)
-[OpenFE + Blending + Explain](https://www.kaggle.com/code/trupologhelper/ps4e5-openfe-blending-explain)
-[S04E05 | Flood Prediction | Ensemble](https://www.kaggle.com/code/ravaghi/s04e05-flood-prediction-ensemble)
- [Flood Prediction Regression LGB XGB CAT [0.86933]](https://www.kaggle.com/code/aspillai/flood-prediction-regression-lgb-xgb-cat-0-86933)
- [PG S4E05 - Cross-Validation madness v1](https://www.kaggle.com/code/suharkov/pg-s4e05-cross-validation-madness-v1)

Thanks for you contributions @trupologhelper ,@suharkov ,@ravaghi ,@aspillai.They were truly helpful!

Apart from that, all others models(XGB,CATBoost,LGBM) were fine tuned with different combination of `grow_policy`,`tree_method`,`objective`,`sampling_method`,etc.

Our best single model score was around `0.86933`,which was not much so I thought ensembling is a way to go.

**- Ensembling:**
The ensemble model used was` LinearRegression()`.
To increase the score further, we created `different subsets of features/models` and determined weights using LinearRegression(), adding them to the pool of OOF predictions.
Finally, we performed `Forward Feature Selection` on all features, including all the engineered subset features.

In this solution, we couldn't get OOF predictions from Autogluon because 
we had less time to get it and also we were facing some version file naming error while loading the autogluon model.
I wish we had OOF predictions from autogluon.
Nevertheless, we did blended it with best solution from @mdoroch (0.86940) which in turn had autogluon in it.

Again, I want to thank @mfmfmf3 and @meloncc for providing high scoring autogluon solution.

Our final ensemble weight was 0.6* Best solution from above + 0.4* @mdoroch solution=`0.86943`,(Private LB:`0.86902`)

@mdoroch solution contained weighted ensemble of his own tuned ensemble of XGB,LGBM,CATBoost and PyBoost and two top public autogluons.

```python
Ensemble weights
ngb                 -0.011743
xgb_params8         -0.058489
lgb_params_bestcv    0.064538
lgb_params_serial   -0.112808
lgb_et_params       -0.024884
lgb_params4         -0.032236
lgb_dart            -0.064403
lgb_params6_goss     0.027780
cat_params_t1       -0.024164
gb_params1          -0.061639
weighted_sum         0.302751
lgbm_oof_preds       0.078312
xgbrf_oof_preds     -0.027127
model3              -0.067031
model5               0.056901
p_xgb_              -0.061300
gamma_xgb           -0.073125
weighted0            0.500579
weighted2            0.122773
weighted3            0.279514
weighted1            0.187362
```
weighted0,weighted1...etc are the weighted ensemble of subsets of features.
ngb(Natural Gradient Boosting),Linear Tree Regression,Linear Forest Regression and Linear Boosting Regression
these are some new model which I used , without tunning and they had small contribution overall.
[Linear Tree](https://github.com/cerlymarco/linear-tree).

Thanks for everyone !!!
