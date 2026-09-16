# #7: A diverse ensemble

Competition: playground-series-s3e10
Rank: #7
Source: https://www.kaggle.com/c/playground-series-s3e10/discussion/396261

First of all: Thanks and congratulations to @paddykb, who came up with a really good generalized additive model (GAM) and published it so that everybody could learn from it.

When I saw @paddykb's notebook, it was clear that I wanted to understand it and then include a GAM in my ensemble. I tried to recreate the GAM in Python, but with modest success - perhaps pygam is a bad implementation, or it's simply my lack of experience. I thus decided to use the original GAM for the ensemble. The challenge was that to determine ensemble weights I needed the oof predictions. This forced my to learn enough R so that I could make @paddykb's code save the oof predictions to a csv file.

Then I applied the principles I was preaching in [this post](https://www.kaggle.com/competitions/playground-series-s3e9/discussion/394592) two weeks ago, optimizing eight other models for best cv score:
- A pipeline of `PolynomialRegression(3)`, logistic regression and `CalibratedClassifierCV` (this is the only model which needed calibration)
- Three `HistGradientBoostingClassifier` with early stopping and soft voting
- CatBoost with `max_depth=3`
- `GradientBoostingClassifier` regularized by `min_samples_leaf=1000`
- LightGBM with a very low learning rate
- XGBoost
- A pipeline of `PolynomialRegression(2)` and XGBoost with `max_depth=2` and a high learning rate
- DART

The tree models were trained with the original dataset, the GAM and logistic regression performed better without the original data.

The correlation matrix shows that the seven tree-based models give similar predictions; only the GAM and the logistic regression deviate from the mainstream:

[Correlation]

Finally I chose ensemble weights which gave a good cv score.

[Final comparison]

[Source code is here](https://www.kaggle.com/code/ambrosm/pss3e10-winning-model).
