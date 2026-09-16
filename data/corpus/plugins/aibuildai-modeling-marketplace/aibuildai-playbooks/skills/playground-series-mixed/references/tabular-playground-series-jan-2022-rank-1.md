# #1 Solution Description: Advanced Linear Model

Competition: tabular-playground-series-jan-2022
Rank: #1
Source: https://www.kaggle.com/c/tabular-playground-series-jan-2022/discussion/304355

The following lines describe the development of my final submission to this competition.

### Importance of cross-validation

In this January TPS one could practice ignoring the public leaderboard. The public leaderboard is based on the first quarter of 2019, but all the interesting holidays occur in April of 2019 or later. This means that the public leaderboard gives no information at all about the quality of a model's holiday features. You can over- or underestimate the influence of Easter, Midsummer Day, National Day, Christmas and so on - for the public leaderboard it doesn't matter. The public leaderboard is only good to verify whether the model deals correctly with the yearly GDP.

For this reason, I focused on cross-validation (`GroupKFold` with the years as groups), and in the cross-validation results, I evaluated the SMAPE for January through March separately from SMAPE for the rest of the year. Then I consistently optimized my model for the latter. For the final evaluation, I submitted the two notebooks with the best cv. The winning notebook has a public lb score of only 4.11991, which would rank it at position 306 of the public lb. It took quite some courage to mark this as the final submission...


### Feature engineering

[My final notebook](https://www.kaggle.com/ambrosm/tpsjan22-10-advanced-linear-model-with-cci) still uses Ridge regression with a log-transformed target, but the features differ from my earlier [linear model](https://www.kaggle.com/ambrosm/tpsjan22-03-linear-model):
- The selection of Fourier coefficients has changed; the stickers get no Fourier coefficients at all (this means that the prediction for the stickers is constant over the whole year).
- There are small changes in the length of holidays.
- The Easter holiday in Norway differs from the Easter holiday in the other two countries.
- I added the OECD's [consumer confidence index](https://www.kaggle.com/ambrosm/oecd-consumer-confidence-index) as external data, as suggested in [this discussion](https://www.kaggle.com/c/tabular-playground-series-jan-2022/discussion/302694).

All these features were found by a detailed analysis of the residuals.

### Why not gradient boosting?

Why didn't I use gradient boosting? The main advantage of gradient boosting in this competition is that it reduces the burden of feature engineering: Decision trees determine automatically which countries and products a holiday affects; linear regression needs manually crafted features. 

The disadvantage is: If you use gradient boosting and don't engineer the features yourself, you give up control. With linear regression you analyze residuals and create an Easter holiday which starts exactly on Good Friday and lasts ten days, with gradient boosting you tune some hyperparameters and accept that the decision trees may find the holiday to last nine or eleven days. If you tune the hyperparameters so that the holiday has exactly ten days, the model will overfit somewhere else.

### Varia

- Regularization of ridge regression is controlled by a single parameter, alpha. It is possible to make the regularization strength depend on the feature by scaling features differently: I did not simply use a `StandardScaler`, but a `ColumnTransformer` with several `MinMaxScaler`s.
- The ratio between KaggleRama and KaggleMart sales is always the same and does not depend on any other features. A direct calculation is more accurate than linear regression.
