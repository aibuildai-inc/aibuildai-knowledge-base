# 26th Place: FE, Pseudo-Labels, Residuals

Competition: playground-series-s5e9
Rank: #26
Source: https://www.kaggle.com/c/playground-series-s5e9/writeups/26th-place-fe-pseudo-labels-residuals

I felt this competition was almost like gambling, but I still learned a lot during the process. I spent a few days experimenting, and my final submission ended up as a blend of top public solutions and my own feature engineering and modeling.
***1. Feature Engineering & Selection***
- Started with raw dataset, performed standard cleaning.
- Selected 54 best features using a combination of:
- Feature importance from XGBoost, LightGBM, CatBoost.
- Permutation importance.
- SHAP values.
***2.Base Models***
- I trained 18 different models  [❤️thanks to Mikhail Naumov, your notebook was a huge inspiration](https://www.kaggle.com/code/mikhailnaumov/beats-per-minute-xgb-lgbm-hgb-nn-ydf), including:
- 6 XGBoost with tuned hyperparameters (Optuna).
- 4 LightGBM.
- 3 HistGradientBoostingRegressor.
- 2 YDFRegressor (GradientBoostedTreesLearner).
- Ridge and ElasticNet.
- Neural Network.

***3. Pseudo-Labeling***
I generated pseudo-labels from the test set:
Calculated residual errors from multiple models.
Selected the lowest-error samples:
```python
thr_mean = np.percentile(abs(residuals_mean), 30)
thr_std  = np.percentile(residuals_std, 30)
```
Chose around 92,804 pseudo-labels (≈30% of test data).
Retrained my models with these pseudo-labels added to training data.
***4. Residual Modeling***
After the first round of training, I trained residual models (stacking) to improve predictions and reduce bias.

***5. Ensembling Strategy***
Final submission was a geometric-like blend of three top public notebooks), plus my own trained model:
Final Prediction
$$
\hat{y} \;=\; 0.5 \cdot \sqrt[3]{\text{sub1} \times \text{sub2} \times \text{sub3}} \;+\; 0.5 \cdot y_{\text{mySub}}
$$


This geometric blend helped stabilize the predictions and slightly improve LB score.
